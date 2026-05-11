#!/usr/bin/env python3
name = "1d_TRIX_VolumeSpike_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1. Load 1d data ONCE for TRIX and volume
    df_1d = get_htf_data(prices, '1d')
    
    # 2. TRIX on daily close
    ema1 = pd.Series(df_1d['close']).ewm(span=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, adjust=False).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix_vals = trix.values
    
    # 3. Align TRIX to daily
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_vals)
    
    # 4. Volume filter: 20-period EMA for spike detection
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.8
    
    # 5. Load 1w data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 6. 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 7. Fixed position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(volume_ok[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        trix_bullish = trix_aligned[i] > 0
        trix_bearish = trix_aligned[i] < 0
        price_above_ema50 = close[i] > ema50_1w_aligned[i]
        price_below_ema50 = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: TRIX positive + price above 1w EMA50 + volume spike
            if trix_bullish and price_above_ema50 and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: TRIX negative + price below 1w EMA50 + volume spike
            elif trix_bearish and price_below_ema50 and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit: TRIX reverses OR price crosses 1w EMA50
            if position == 1:
                if trix_aligned[i] < 0 or close[i] < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if trix_aligned[i] > 0 or close[i] > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals