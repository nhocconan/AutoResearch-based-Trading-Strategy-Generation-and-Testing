#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1w pivot levels with breakout and fade logic
# - Fade (counter-trend) at S2/R2 levels when price reverses from these levels with volume confirmation
# - Breakout (trend-following) at S3/R3 levels when price breaks through with volume expansion
# - Uses 1w pivot levels calculated from prior weekly bar's range
# - Adds 1d EMA50 filter to only take trades in direction of daily trend
# - Designed to work in ranging markets (fades at S2/R2) and trending markets (breakouts at S3/R3)
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing
# - Focus on BTC/ETH with SOL as secondary confirmation

name = "4h_1wPivot_S2R2_S3R3_1dEMA50_Trend_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot levels: based on prior weekly bar's range
    # R3 = high + 2*(high - low), R2 = pivot + (high - low)
    # S2 = pivot - (high - low), S3 = low - 2*(high - low)
    # pivot = (high + low + close) / 3
    prev_high = df_1w['high'].shift(1)
    prev_low = df_1w['low'].shift(1)
    prev_close = df_1w['close'].shift(1)
    
    pivot = (prev_high + prev_low + prev_close) / 3
    price_range = prev_high - prev_low
    
    R3 = prev_high + 2 * price_range
    R2 = pivot + price_range
    S2 = pivot - price_range
    S3 = prev_low - 2 * price_range
    
    # Align weekly pivot levels to 4h timeframe
    R3_4h = align_htf_to_ltf(prices, df_1w, R3.values)
    R2_4h = align_htf_to_ltf(prices, df_1w, R2.values)
    S2_4h = align_htf_to_ltf(prices, df_1w, S2.values)
    S3_4h = align_htf_to_ltf(prices, df_1w, S3.values)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filters
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)  # Volume confirmation
    volume_expansion = volume > np.roll(volume, 1)  # Current volume > previous
    volume_expansion[0] = False
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(R3_4h[i]) or np.isnan(R2_4h[i]) or np.isnan(S2_4h[i]) or np.isnan(S3_4h[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i]) or np.isnan(volume_expansion[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Fade at S2/R2: counter-trend entry when price reverses from these levels
            # Long fade at S2: price touches S2 and closes back above it
            if close[i] <= S2_4h[i] * 1.001 and close[i] > S2_4h[i] and volume_filter[i]:
                # Only take long fade if below daily EMA (bearish context)
                if close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short fade at R2: price touches R2 and closes back below it
            elif close[i] >= R2_4h[i] * 0.999 and close[i] < R2_4h[i] and volume_filter[i]:
                # Only take short fade if above daily EMA (bullish context)
                if close[i] > ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            # Breakout at S3/R3: trend-following entry when price breaks through
            # Long breakout at R3: price breaks above R3 with volume expansion
            elif close[i] > R3_4h[i] and volume_expansion[i]:
                # Only take long breakout if above daily EMA (bullish context)
                if close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakout at S3: price breaks below S3 with volume expansion
            elif close[i] < S3_4h[i] and volume_expansion[i]:
                # Only take short breakout if below daily EMA (bearish context)
                if close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price reaches R2 (profit target) or breaks below S3 (stop)
            if close[i] >= R2_4h[i] * 0.999:  # Take profit at R2
                signals[i] = 0.0
                position = 0
            elif close[i] < S3_4h[i]:  # Stop loss if breaks below S3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches S2 (profit target) or breaks above R3 (stop)
            if close[i] <= S2_4h[i] * 1.001:  # Take profit at S2
                signals[i] = 0.0
                position = 0
            elif close[i] > R3_4h[i]:  # Stop loss if breaks above R3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals