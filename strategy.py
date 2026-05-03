#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability intraday reversal points.
# Breakout above R3 or below S3 with 1d EMA34 trend alignment captures strong momentum moves.
# Volume spike confirms institutional participation. Designed for 12-37 trades/year on 12h to minimize fee drag.
# Works in bull markets via buying R3 breakouts in uptrends and in bear markets via selling S3 breakdowns in downtrends.

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get previous day's high, low, close for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    camarilla_r3 = prev_close + range_val * 1.1 / 4
    camarilla_s3 = prev_close - range_val * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous day data
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA
        if i >= 19:
            vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_r3[i]
        breakout_down = close[i] < camarilla_s3[i]
        
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
            # Exit long: price returns to Camarilla H3/L3 level or loses 1d uptrend
            camarilla_h3 = prev_close[i] + range_val[i] * 1.1 / 6
            camarilla_l3 = prev_close[i] - range_val[i] * 1.1 / 6
            if close[i] < camarilla_h3 or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla H3/L3 level or loses 1d downtrend
            camarilla_h3 = prev_close[i] + range_val[i] * 1.1 / 6
            camarilla_l3 = prev_close[i] - range_val[i] * 1.1 / 6
            if close[i] > camarilla_l3 or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals