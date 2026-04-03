#!/usr/bin/env python3
"""
Experiment #1219: 6h Elder Ray + 12h Trend Regime + Volume Spike
HYPOTHESIS: Elder Ray (Bull/Bear Power) on 6h captures short-term momentum with mean-reversion tendency in ranging markets. 
Combined with 12h trend regime (price vs EMA50) to filter counter-trend trades, and volume spike (>2x average) to ensure 
institutional participation. In bull regimes (price>EMA50), we take Bull Power signals; in bear regimes (price<EMA50), 
we take Bear Power signals. This adapts to both trending and ranging markets while minimizing false breakouts. 
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1219_6h_elder_ray_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend regime filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # EMA50 on 12h for trend regime
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    # Trend regime: 1 = bull (price > EMA50), -1 = bear (price < EMA50)
    trend_regime_12h = np.where(close_12h > ema_50_12h, 1, -1)
    trend_regime_aligned = align_htf_to_ltf(prices, df_12h, trend_regime_12h)
    
    # === 6h Indicators: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) ===
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13  # Bull Power: measures bullish strength
    bear_power = low - ema_13   # Bear Power: measures bearish strength (negative values)
    
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
    
    warmup = max(20, 13, 50)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_13[i]) or np.isnan(trend_regime_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # In bull regime (12h price > EMA50): look for Bull Power signals
            if trend_regime_aligned[i] > 0:  # 12h bull regime
                # Bull Power > 0 and rising (bullish momentum)
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            # In bear regime (12h price < EMA50): look for Bear Power signals
            else:  # 12h bear regime
                # Bear Power < 0 and falling (increasing bearish momentum)
                if bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            # No signal conditions met
            if not in_position:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals