# 4h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Combines Camarilla pivot levels (R3/S3) from daily timeframe with daily trend filter (EMA34) and volume spike.
# Camarilla levels provide high-probability reversal/breakout zones; daily trend ensures alignment with higher timeframe bias.
# Volume spike confirms institutional participation. Designed for fewer, high-quality trades in both bull and bear markets.
# Target: 20-50 trades/year to avoid fee drag. Works in ranging markets (reversion at R3/S3) and trending markets (breakouts).

#!/usr/bin/env python3
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
    
    # Get daily data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily close, high, low for Camarilla
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels for each day: R3, S3
    # R3 = close + 1.1*(high - low)/6
    # S3 = close - 1.1*(high - low)/6
    rang = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * rang / 6
    camarilla_s3 = close_1d - 1.1 * rang / 6
    
    # Align to 4h timeframe (wait for daily close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need EMA34 and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema_trend = ema_34_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above R3 in uptrend + volume spike
            if close[i] > r3_level and close[i] > ema_trend and vol_ok:
                signals[i] = size
                position = 1
            # Short: price breaks below S3 in downtrend + volume spike
            elif close[i] < s3_level and close[i] < ema_trend and vol_ok:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price re-enters below R3 or trend reverses
            if close[i] < r3_level or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price re-enters above S3 or trend reverses
            if close[i] > s3_level or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0