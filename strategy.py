#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability intraday support/resistance. 
# Breakouts above R3 or below S3 with 12h EMA34 trend alignment and volume spike capture strong momentum moves.
# Designed for 20-50 trades/year on 4h to minimize fee drag while maintaining edge in both bull and bear markets.

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_VolumeSpike"
timeframe = "4h"
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
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Need at least 20 bars for Camarilla calculation
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_12h_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla levels (based on previous day's range)
        if i >= 24:  # Need at least 24 hours (96 4h bars) for proper daily calculation, but approximate with 20-bar lookback
            # Use 20-bar lookback as proxy for daily range (more responsive)
            lookback = min(20, i)
            highest_high = np.max(high[i-lookback:i])
            lowest_low = np.min(low[i-lookback:i])
            close_prev = close[i-1]
            
            # Calculate Camarilla levels
            range_val = highest_high - lowest_low
            if range_val > 0:
                camarilla_r3 = close_prev + (range_val * 1.1 / 4)
                camarilla_s3 = close_prev - (range_val * 1.1 / 4)
            else:
                camarilla_r3 = close_prev
                camarilla_s3 = close_prev
        else:
            # Not enough data yet
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA on 4h
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        if position == 0:
            # Long: price breaks above Camarilla R3 in 12h uptrend with volume spike
            if close[i] > camarilla_r3 and ema_34_12h_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 in 12h downtrend with volume spike
            elif close[i] < camarilla_s3 and ema_34_12h_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla R3 or loses 12h uptrend
            if close[i] < camarilla_r3 or ema_34_12h_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla S3 or loses 12h downtrend
            if close[i] > camarilla_s3 or ema_34_12h_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals