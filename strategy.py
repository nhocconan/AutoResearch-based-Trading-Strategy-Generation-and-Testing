#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_cam_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily OHLC for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels
    # Resistance levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Shift by 1 to use only completed daily bars (previous day's levels)
    pivot_1d = np.roll(pivot_1d, 1)
    r1_1d = np.roll(r1_1d, 1)
    r2_1d = np.roll(r2_1d, 1)
    r3_1d = np.roll(r3_1d, 1)
    r4_1d = np.roll(r4_1d, 1)
    s1_1d = np.roll(s1_1d, 1)
    s2_1d = np.roll(s2_1d, 1)
    s3_1d = np.roll(s3_1d, 1)
    s4_1d = np.roll(s4_1d, 1)
    pivot_1d[0] = np.nan
    r1_1d[0] = np.nan
    r2_1d[0] = np.nan
    r3_1d[0] = np.nan
    r4_1d[0] = np.nan
    s1_1d[0] = np.nan
    s2_1d[0] = np.nan
    s3_1d[0] = np.nan
    s4_1d[0] = np.nan
    
    # Align daily Camarilla levels to 1h timeframe
    pivot_1h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1h = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1h = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1h = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1h = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1h ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(300, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_1h[i]) or np.isnan(r1_1h[i]) or np.isnan(r2_1h[i]) or np.isnan(r3_1h[i]) or np.isnan(r4_1h[i]) or
            np.isnan(s1_1h[i]) or np.isnan(s2_1h[i]) or np.isnan(s3_1h[i]) or np.isnan(s4_1h[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.5x average)
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long conditions: price breaks above S3 with volume and in session
        long_signal = volume_confirmed and in_session and (price_high > s3_1h[i])
        
        # Short conditions: price breaks below R3 with volume and in session
        short_signal = volume_confirmed and in_session and (price_low < r3_1h[i])
        
        # Exit when price returns to pivot (mean reversion)
        exit_long = position == 1 and price_close < pivot_1h[i]
        exit_short = position == -1 and price_close > pivot_1h[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1h Camarilla breakout strategy with daily pivot levels, volume confirmation, and session filter (08-20 UTC).
# Enters long when 1h price breaks above daily S3 level (close - 1.1*(H-L)/4) with volume >1.5x average and during active session.
# Enters short when price breaks below daily R3 level (close + 1.1*(H-L)/4) with same conditions.
# Exits when price returns to the daily pivot point (mean reversion within the day's range).
# Uses daily Camarilla levels for institutional support/resistance, volume to confirm breakout strength,
# and session filter to avoid low-liquidity periods. Target: 20-40 trades/year to minimize fee drag while capturing
# meaningful intraday moves aligned with daily structure. Works in both bull and bear markets by fading false breaks
# and capturing genuine intraday trends that respect daily pivot levels.