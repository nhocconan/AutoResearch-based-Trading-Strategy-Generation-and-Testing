#!/usr/bin/env python3
"""
Experiment #3835: 6h Williams %R + 1w EMA50 trend filter + volume spike
HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions with mean reversion tendency. 
1w EMA50 defines the primary trend - only take longs above EMA50, shorts below EMA50 to avoid fighting trend.
Volume spike (>2.0x) confirms institutional participation at turning points.
Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).
Discrete position sizing (0.25) minimizes fee drag. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3835_6h_williamsr_1w_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for EMA50 trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1w EMA50 to 6h timeframe (shifted by 1 for completed 1w bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === 6h Indicators: Williams %R(14) ===
    lookback_wr = 14
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = np.full(n, np.nan)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    mask = (highest_high - lowest_low) != 0
    williams_r[mask] = ((highest_high[mask] - close[mask]) / (highest_high[mask] - lowest_low[mask])) * -100
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(lookback_wr + 1, 20, 50)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(williams_r[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema50 = ema50_1w_aligned[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit conditions: Williams %R returns to neutral zone (-50) or opposite extreme
            if position_side > 0:  # Long
                # Exit if Williams %R rises above -50 (returning from oversold)
                # or if price breaks below 6h EMA20 (trend change)
                if williams_r[i] > -50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit if Williams %R falls below -50 (returning from overbought)
                # or if price breaks above 6h EMA20 (trend change)
                if williams_r[i] < -50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: Williams %R oversold (< -80) AND price above 1w EMA50 (uptrend)
            if (williams_r[i] < -80 and  # Oversold condition
                price > ema50):          # Above weekly EMA50 (uptrend filter)
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: Williams %R overbought (> -20) AND price below 1w EMA50 (downtrend)
            elif (williams_r[i] > -20 and   # Overbought condition
                  price < ema50):           # Below weekly EMA50 (downtrend filter)
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals