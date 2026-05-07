#!/usr/bin/env python3
name = "1d_Wedges_Breakout_WeeklyTrend_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily EMAs for wedge detection
    ema_9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly EMA for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(ema_9[i]) or 
            np.isnan(ema_20[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Rising wedge (bearish): higher lows + lower highs
        rising_wedge = (ema_9[i] > ema_9[i-1]) and (ema_20[i] < ema_20[i-1])
        # Falling wedge (bullish): lower lows + higher highs
        falling_wedge = (ema_9[i] < ema_9[i-1]) and (ema_20[i] > ema_20[i-1])
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: falling wedge breakout above EMA20 in weekly uptrend
            if falling_wedge and close[i] > ema_20[i] and vol_condition and ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: rising wedge breakdown below EMA20 in weekly downtrend
            elif rising_wedge and close[i] < ema_20[i] and vol_condition and ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below EMA9 or wedge invalid
            if close[i] < ema_9[i] or not falling_wedge:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above EMA9 or wedge invalid
            if close[i] > ema_9[i] or not rising_wedge:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 1d Wedge breakouts with weekly trend filter and volume confirmation
# - Rising wedge (higher lows + lower highs) = bearish continuation pattern
# - Falling wedge (lower lows + higher highs) = bullish continuation pattern
# - Breakout occurs when price breaks EMA20 in wedge direction with volume spike
# - Weekly EMA20 trend filter ensures alignment with higher timeframe trend
# - Works in both bull (falling wedge breakouts in uptrend) and bear (rising wedge breakdowns in downtrend)
# - Volume confirmation (2x average) reduces false breakouts
# - Exit when price returns to EMA9 or wedge pattern invalidates
# - Position size 0.25 targets ~30-80 trades/year to avoid fee drag
# - Wedges provide clear structure with defined support/resistance levels
# - Weekly trend filter reduces whipsaws vs same-timeframe signals
# - Novel combination: Wedge patterns (9/20 EMA) + weekly trend + volume spike not recently tried
# - Aims for 60-120 total trades over 4 years (15-30/year) to stay within limits