#!/usr/bin/env python3
"""
1d_Keltner_Breakout_1wTrend_Volume
Hypothesis: Keltner Channel breakout on daily timeframe with weekly trend filter and volume confirmation.
Works in bull: price breaks above upper Keltner + weekly uptrend + volume spike.
Works in bear: price breaks below lower Keltner + weekly downtrend + volume spike.
Designed for 10-30 trades per year on 1d timeframe with low turnover.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Keltner Channel parameters
    kc_period = 20
    kc_mult = 2.0
    
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR
    atr = pd.Series(tr).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    
    # EMA of close
    ema_close = pd.Series(close).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    
    # Keltner Bands
    kc_upper = ema_close + (kc_mult * atr)
    kc_lower = ema_close - (kc_mult * atr)
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for calculations
    start_idx = kc_period + 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr[i]) or np.isnan(ema_close[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_20_1w_val = ema_20_1w_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: close above upper KC AND weekly uptrend AND volume spike
            if close[i] > kc_upper[i] and ema_20_1w_val > close_1w[-1] and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: close below lower KC AND weekly downtrend AND volume spike
            elif close[i] < kc_lower[i] and ema_20_1w_val < close_1w[-1] and vol_spike_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: close below EMA (middle) OR weekly trend turns down
            if close[i] < ema_close[i] or ema_20_1w_val < close_1w[-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above EMA (middle) OR weekly trend turns up
            if close[i] > ema_close[i] or ema_20_1w_val > close_1w[-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Keltner_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0