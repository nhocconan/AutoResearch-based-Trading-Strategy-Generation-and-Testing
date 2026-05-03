#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels identify key intraday support/resistance. Breakout above R3 or below S3
# with volume confirmation suggests institutional participation. 1d EMA34 filter ensures trades
# align with the daily trend to avoid counter-trend whipsaws. Designed for 12-30 trades/year
# on 6h to minimize fee drag while capturing sustained moves in both bull and bear markets.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous day data
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get previous day's OHLC for Camarilla calculation
        if i-1 < len(df_1d):
            prev_high = df_1d['high'].iloc[i-1]
            prev_low = df_1d['low'].iloc[i-1]
            prev_close = df_1d['close'].iloc[i-1]
        else:
            # Not enough 1d data yet
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            # Avoid division by zero
            r3 = s3 = prev_close
            r4 = s4 = prev_close
        else:
            r3 = prev_close + range_val * 1.1 / 4
            s3 = prev_close - range_val * 1.1 / 4
            r4 = prev_close + range_val * 1.1 / 2
            s4 = prev_close - range_val * 1.1 / 2
        
        # Volume confirmation: 20-period EMA
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (2.0 * vol_ema_20)
        
        if position == 0:
            # Long: break above R3 with volume spike in 1d uptrend
            if close[i] > r3 and volume_spike and ema_34_1d_aligned[i] > close[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume spike in 1d downtrend
            elif close[i] < s3 and volume_spike and ema_34_1d_aligned[i] < close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 or loses 1d uptrend
            if close[i] < r3 or ema_34_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 or loses 1d downtrend
            if close[i] > s3 or ema_34_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals