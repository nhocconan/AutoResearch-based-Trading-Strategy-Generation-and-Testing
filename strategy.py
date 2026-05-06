#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and ATR volatility filter
# Uses Camarilla pivot levels for structure, 4h EMA50 for trend alignment (reduces whipsaw)
# ATR(14) > 20-bar average ATR filters for sufficient volatility to avoid choppy markets
# Discrete sizing 0.20 to limit fee drag; target 60-150 total trades over 4 years
# Uses 4h/1d for signal direction, 1h only for entry timing
# Session filter 08-20 UTC to reduce noise trades
# Proven pattern: price channel breakouts with volume/volatility confirmation work on BTC/ETH in both bull/bear

name = "1h_Camarilla_R3S3_4hEMA50_ATRFilter_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h EMA50 trend filter
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use R3 and S3 for breakouts
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for today's Camarilla levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First value has no previous day
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla R3 and S3
    camarilla_range = prev_high - prev_low
    r3 = prev_close + (camarilla_range * 1.1 / 4)
    s3 = prev_close - (camarilla_range * 1.1 / 4)
    
    # Calculate ATR(14) for volatility filter on 4h
    tr1 = pd.Series(high_4h).shift(1) - pd.Series(low_4h).shift(1)
    tr2 = abs(pd.Series(high_4h).shift(1) - pd.Series(close_4h).shift(1))
    tr3 = abs(pd.Series(low_4h).shift(1) - pd.Series(close_4h).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    avg_atr_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr > (1.2 * avg_atr_20)  # Require above-average volatility
    
    # Align HTF indicators to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    volatility_filter_aligned = align_htf_to_ltf(prices, df_4h, volatility_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volatility_filter_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3 AND uptrend (price > EMA50) AND sufficient volatility
            if close[i] > r3_aligned[i] and close[i] > ema50_4h_aligned[i] and volatility_filter_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short breakdown: price < S3 AND downtrend (price < EMA50) AND sufficient volatility
            elif close[i] < s3_aligned[i] and close[i] < ema50_4h_aligned[i] and volatility_filter_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price retests S3 from above (trend reversal)
            if close[i] <= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price retests R3 from below (trend reversal)
            if close[i] >= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals