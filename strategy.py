#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. In ranging markets, mean reversion
# from extreme levels works well. Trend filter ensures we trade with the weekly trend.
# Volume confirmation reduces false signals. Designed for 20-40 trades/year on 1d to
# minimize fee drag while maintaining edge in both bull and bear markets.

name = "1d_WilliamsR_MeanReversion_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after sufficient warmup for Williams %R
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R conditions with volume confirmation
        vol_confirm = volume[i] > vol_ema_20[i]
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        if position == 0:
            # Long: oversold in weekly uptrend with volume confirmation
            if oversold and ema_50_1w_aligned[i] > close[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: overbought in weekly downtrend with volume confirmation
            elif overbought and ema_50_1w_aligned[i] < close[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above -50 (middle) or loses weekly uptrend
            if williams_r[i] > -50 or ema_50_1w_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below -50 (middle) or loses weekly downtrend
            if williams_r[i] < -50 or ema_50_1w_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals