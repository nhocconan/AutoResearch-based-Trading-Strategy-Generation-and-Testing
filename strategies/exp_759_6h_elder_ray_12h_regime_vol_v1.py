#!/usr/bin/env python3
"""
Experiment #759: 6h Elder Ray + 12h Regime Filter + Volume Spike
HYPOTHESIS: Elder Ray (Bull Power/Bear Power) captures bull/bear strength, filtered by 12h ADX regime (ADX>25 = trending) 
and volume confirmation (>1.5x average). Long when Bull Power > 0 AND Bear Power < 0 (bullish dominance) in trending regime. 
Short when Bear Power > 0 AND Bull Power < 0 (bearish dominance) in trending regime. Works in bull/bear markets: 
in bull trends, Elder Ray shows bullish dominance; in bear trends, shows bearish dominance. Uses discrete position sizing (0.25).
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_759_6h_elder_ray_12h_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX regime filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / np.where(atr == 0, 1, atr)
        di_minus = 100 * dm_minus_smooth / np.where(atr == 0, 1, atr)
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    # Regime: 1 = trending (ADX > 25), 0 = ranging (ADX <= 25)
    regime_12h = np.where(adx_12h > 25, 1, 0)
    # Align regime to 6h timeframe
    regime_12h_aligned = align_htf_to_ltf(prices, df_12h, regime_12h)
    
    # === 6h Indicators: Elder Ray (Bull Power, Bear Power) ===
    # EMA(13) as proxy for market consensus
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 13, 14)  # sufficient for volume MA, EMA13, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(regime_12h_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~48h on 6h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike and regime_12h_aligned[i] > 0:  # Only trade in trending regime
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish dominance)
            if bull_power[i] > 0 and bear_power[i] < 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Bear Power > 0 AND Bull Power < 0 (bearish dominance)
            elif bear_power[i] > 0 and bull_power[i] < 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals