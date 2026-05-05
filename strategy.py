#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation
# Long when price breaks above 1h Camarilla R3 AND price > 4h EMA34 (uptrend) AND volume > 2.0x 24-period average
# Short when price breaks below 1h Camarilla S3 AND price < 4h EMA34 (downtrend) AND volume > 2.0x 24-period average
# Exit when price crosses 1h Camarilla midpoint (M3) OR 4h EMA34 filter reverses
# Uses Camarilla pivots for precise intraday structure, 4h EMA34 for regime filter (avoid whipsaws in chop)
# Volume spike confirms institutional participation
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# Timeframe: 1h (as required)
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
# Session filter: 08-20 UTC to reduce noise trades

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_VolumeSpike"
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
    
    # Get 1h data ONCE before loop for Camarilla calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 5:
        return np.zeros(n)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate 1h Camarilla levels (R3, S3, M3)
    # Pivot = (H + L + C) / 3
    # R3 = H + 2*(C - L) / 2 = H + (C - L) = H + C - L
    # S3 = L - 2*(H - C) / 2 = L - (H - C) = L - H + C
    # M3 = (R3 + S3) / 2
    pivot_1h = (high_1h + low_1h + close_1h) / 3.0
    r3_1h = high_1h + (close_1h - low_1h)
    s3_1h = low_1h - (high_1h - close_1h)
    m3_1h = (r3_1h + s3_1h) / 2.0
    
    # Get 4h data ONCE before loop for EMA34
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(34)
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1h, r3_1h)
    s3_aligned = align_htf_to_ltf(prices, df_1h, s3_1h)
    m3_aligned = align_htf_to_ltf(prices, df_1h, m3_1h)
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation on 1h (threshold: 2.0x)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_spike = volume > (2.0 * vol_ma_24)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(m3_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND price > EMA34 (uptrend) AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND price < EMA34 (downtrend) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below M3 OR price < EMA34 (trend weakening)
            if close[i] < m3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above M3 OR price > EMA34 (trend weakening)
            if close[i] > m3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals