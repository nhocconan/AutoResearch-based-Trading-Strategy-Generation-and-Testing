#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R with 1w trend filter and volume confirmation
# - Williams %R(14) from daily timeframe for overbought/oversold signals
# - Only take long when %R < -80 (oversold) and 1w EMA50 uptrend
# - Only take short when %R > -20 (overbought) and 1w EMA50 downtrend
# - Volume confirmation: current volume > 1.5x 20-period average
# - Session filter: 08-20 UTC to avoid low liquidity periods
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing
# - Williams %R is effective in ranging markets and captures reversals in trends
# - Weekly EMA filter ensures we trade with the higher timeframe trend

name = "6h_WilliamsR_1d_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    rr = np.where(rr == 0, 0.0001, rr)
    williams_r = ((highest_high - df_1d['close'].values) / rr) * -100
    
    # Align Williams %R to 6h timeframe
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filters
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(williams_r_6h[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long signal: Williams %R oversold (< -80) and 1w uptrend (price > EMA50)
            if williams_r_6h[i] < -80 and close[i] > ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short signal: Williams %R overbought (> -20) and 1w downtrend (price < EMA50)
            elif williams_r_6h[i] > -20 and close[i] < ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R reaches overbought territory or trend reversal
            if williams_r_6h[i] > -20 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R reaches oversold territory or trend reversal
            if williams_r_6h[i] < -80 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals