#!/usr/bin/env python3
"""
Experiment #131: 6h Elder Ray + 1d Regime Filter + Volume Spike
HYPOTHESIS: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength. 
Combine with 1d regime filter (ADX > 25 = trend, ADX < 20 = range) and volume confirmation (>1.5x) 
to capture strong trends while avoiding whipsaws. In trending regimes (ADX>25): go long when Bull Power > 0 and rising, 
short when Bear Power > 0 and rising. In ranging regimes (ADX<20): fade extremes (long when Bear Power < -0.5*ATR, 
short when Bull Power < -0.5*ATR). Volume spike ensures institutional participation. 
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_131_6h_elder_ray_1d_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # Calculate ADX(14) on 1d
    def calculate_adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx.values
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_regime = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: EMA(13) for Elder Ray ===
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === 6h Indicators: Elder Ray ===
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = ema_13 - low   # Bear Power = EMA - Low
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for EMA13 + ADX warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(adx_regime[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Regime Detection ---
        is_trending = adx_regime[i] > 25
        is_ranging = adx_regime[i] < 20
        
        # --- Elder Ray Signals ---
        # Bull Power rising (current > previous)
        bull_power_rising = bull_power[i] > bull_power[i-1] if i > 0 else False
        # Bear Power rising (current > previous)
        bear_power_rising = bear_power[i] > bear_power[i-1] if i > 0 else False
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~48h on 6h)
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            if is_trending:
                # In trending regime: follow Elder Ray momentum
                # Long: Bull Power > 0 and rising
                if bull_power[i] > 0 and bull_power_rising:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short: Bear Power > 0 and rising
                elif bear_power[i] > 0 and bear_power_rising:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # In ranging regime: fade extremes
                # Long: Bear Power strongly negative (oversold)
                if bear_power[i] < -0.5 * atr[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short: Bull Power strongly negative (overbought)
                elif bull_power[i] < -0.5 * atr[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                # Transition regime (ADX 20-25): no trade
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals