#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivots provide intraday support/resistance levels effective in ranging and trending markets.
# 4h EMA50 ensures alignment with intermediate trend to avoid counter-trend entries.
# Volume confirmation filters false breakouts. Session filter (08-20 UTC) reduces noise.
# Designed for 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
# Works in bull markets via upward breaks at R3/R4 and in bear markets via downward breaks at S3/S4.

name = "1h_Camarilla_R3S3_4hEMA50_VolumeSpike"
timeframe = "1h"
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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d data for Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # where C = (H+L+Close)/3 (typical price)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    hl_range_1d = df_1d['high'] - df_1d['low']
    
    r4_1d = typical_price_1d + (hl_range_1d * 1.1 / 2.0)
    r3_1d = typical_price_1d + (hl_range_1d * 1.1 / 4.0)
    s3_1d = typical_price_1d - (hl_range_1d * 1.1 / 4.0)
    s4_1d = typical_price_1d - (hl_range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 1h timeframe (using previous day's levels)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d.values)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d.values)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d.values)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d.values)
    
    # Volume confirmation: 24-period EMA on 1h (equivalent to 12 periods on 2h, but we use 24 for 1h)
    vol_ema_24 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_24_values = vol_series.ewm(span=24, adjust=False, min_periods=24).mean().values
    vol_ema_24[:] = vol_ema_24_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # Start from 24 to have valid volume EMA and aligned HTF data
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(vol_ema_24[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_24[i])
        
        if position == 0:
            # Long: price breaks above R3 with uptrend alignment and volume spike
            if close[i] > r3_1d_aligned[i] and ema_50_4h_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 with downtrend alignment and volume spike
            elif close[i] < s3_1d_aligned[i] and ema_50_4h_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or loses uptrend alignment
            if close[i] < s3_1d_aligned[i] or ema_50_4h_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R3 or loses downtrend alignment
            if close[i] > r3_1d_aligned[i] or ema_50_4h_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals