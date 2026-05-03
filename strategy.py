#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla levels from prior day provide precise support/resistance.
# Breakout above R3 (long) or below S3 (short) with 1d trend alignment and volume spike
# captures institutional momentum with low false breaks. Designed for 20-40 trades/year on 4h.
# Works in bull markets via buying R3 breakouts in uptrends and bear markets via selling S3 breakdowns in downtrends.
# Uses 1d EMA34 for trend filter to avoid calling get_htf_data inside loop.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for trend filter and Camarilla calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute 20-period volume EMA on 4h for efficiency
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=1).mean().values
    
    for i in range(6, n):  # Need at least 6*4h = 24h to form prior day
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Need prior day's completed OHLC for Camarilla (use second-to-last daily bar)
        if len(df_1d) >= 2:
            prior_high = df_1d['high'].iloc[-2]
            prior_low = df_1d['low'].iloc[-2]
            prior_close = df_1d['close'].iloc[-2]
        else:
            # Not enough prior days yet
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla R3 and S3 levels
        range_val = prior_high - prior_low
        if range_val <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3 = prior_close + (range_val * 1.1 / 4)
        s3 = prior_close - (range_val * 1.1 / 4)
        
        # Volume confirmation: current volume > 1.5 * 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Camarilla breakout conditions
        breakout_up = close[i] > r3
        breakout_down = close[i] < s3
        
        if position == 0:
            # Long: bullish breakout above R3 in 1d uptrend with volume spike
            if breakout_up and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakdown below S3 in 1d downtrend with volume spike
            elif breakout_down and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to R3 level or loses 1d uptrend
            if close[i] < r3 or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to S3 level or loses 1d downtrend
            if close[i] > s3 or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals