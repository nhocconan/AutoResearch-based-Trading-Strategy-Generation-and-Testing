#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla Pivot (R3/S3) breakout with 1w EMA34 trend filter and volume confirmation
# Camarilla pivots identify key intraday support/resistance levels. Breakouts above R3 or below S3
# with 1w EMA34 trend alignment capture strong momentum moves. Volume spike confirms conviction.
# Designed for 1d timeframe to target 30-100 trades over 4 years (7-25/year) minimizing fee drag.
# Works in both bull and bear markets by trading breakouts in the direction of higher timeframe trend.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # Need at least 2 days for pivot calculation
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1w_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla pivot levels for current day using previous day's OHLC
        if i < 2:
            continue  # Need at least 2 days of data
            
        # Previous day's OHLC
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3.0
        
        # Calculate Camarilla levels (R3, S3)
        range_val = prev_high - prev_low
        r3 = pivot + (range_val * 1.1 / 4.0)  # R3 = pivot + (high-low)*1.1/4
        s3 = pivot - (range_val * 1.1 / 4.0)  # S3 = pivot - (high-low)*1.1/4
        
        # Volume confirmation: 20-period EMA
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Breakout conditions
        breakout_long = close[i] > r3  # Price breaks above R3
        breakout_short = close[i] < s3  # Price breaks below S3
        
        if position == 0:
            # Long: breakout above R3 in 1w uptrend with volume spike
            if breakout_long and ema_34_1w_aligned[i] > close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 in 1w downtrend with volume spike
            elif breakout_short and ema_34_1w_aligned[i] < close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below pivot or loses 1w uptrend
            if close[i] < pivot or ema_34_1w_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above pivot or loses 1w downtrend
            if close[i] > pivot or ema_34_1w_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals