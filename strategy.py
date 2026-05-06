#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Camarilla pivot levels with volume spike and trend filter
# Long when price breaks above R3 with volume > 2x average and price above 1-day EMA34 (bullish trend)
# Short when price breaks below S3 with volume > 2x average and price below 1-day EMA34 (bearish trend)
# Camarilla levels provide precise intraday support/resistance. Volume confirms breakout strength.
# EMA34 filter ensures trades align with higher timeframe trend, reducing false breakouts.
# Works in bull/bear markets: breakouts capture momentum, trend filter avoids counter-trend trades.
# Target: 20-50 trades per year (80-200 over 4 years) with 0.25 position sizing.

name = "4h_1dCamarilla_R3S3_TrendVol_Breakout_v1"
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
    
    # Calculate 1-day Camarilla levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla pivot levels
    range_ = prev_high - prev_low
    pivot = (prev_high + prev_low + prev_close) / 3
    
    # Resistance levels
    r1 = pivot + (range_ * 1.1 / 12)
    r2 = pivot + (range_ * 1.1 / 6)
    r3 = pivot + (range_ * 1.1 / 4)
    
    # Support levels
    s1 = pivot - (range_ * 1.1 / 12)
    s2 = pivot - (range_ * 1.1 / 6)
    s3 = pivot - (range_ * 1.1 / 4)
    
    # Align 1-day levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: >2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R3 with volume and trend confirmation
            if close[i] > r3_aligned[i] and volume_filter[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S3 with volume and trend confirmation
            elif close[i] < s3_aligned[i] and volume_filter[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (failed support) or trend turns bearish
            if close[i] < s3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 (failed resistance) or trend turns bullish
            if close[i] > r3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals