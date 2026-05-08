#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1-week trend filter and volume confirmation
# Long when Williams %R crosses above -20 (oversold bounce) + weekly EMA(20) uptrend + volume spike
# Short when Williams %R crosses below -80 (overbought rejection) + weekly EMA(20) downtrend + volume spike
# Williams %R identifies overbought/oversold conditions effective in ranging and trending markets
# Weekly trend filter ensures alignment with higher timeframe momentum
# Volume spike confirms institutional participation
# Targets 30-100 total trades over 4 years (7-25/year) to avoid fee drag

name = "1d_WilliamsR_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    weekly_close = df_1w['close'].values
    ema20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_1w_val = ema20_1w_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Williams %R crosses above -20 + weekly uptrend + volume spike
            if i > start_idx and williams_r[i-1] <= -20 and wr > -20 and close[i] > ema20_1w_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -80 + weekly downtrend + volume spike
            elif i > start_idx and williams_r[i-1] >= -80 and wr < -80 and close[i] < ema20_1w_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -80 OR weekly trend turns down
            if i > start_idx and williams_r[i-1] >= -80 and wr < -80:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -20 OR weekly trend turns up
            if i > start_idx and williams_r[i-1] <= -20 and wr > -20:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals