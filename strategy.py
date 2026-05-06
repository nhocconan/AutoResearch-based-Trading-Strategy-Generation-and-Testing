# 1. Hypothesis: 4h strategy using daily Pivot Points (High, Low, Close of prior day) with breakout and fade logic
# - Fade (counter-trend) at S1/R1 levels when price reverses from these levels with volume confirmation
# - Breakout (trend-following) at S2/R2 levels when price breaks through with volume expansion
# - Uses daily Pivot Points calculated from prior day's OHLC
# - Adds 4h EMA50 filter to only take trades in direction of 4h trend
# - Designed to work in ranging markets (fades at S1/R1) and trending markets (breakouts at S2/R2)
# - Target: 75-200 total trades over 4 years (19-50/year) with 0.25 position sizing
# - BTC/ETH focus: Pivot points work well in ranging/mean-reverting markets (bear) and breakout markets (bull)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_DailyPivot_S1R1_S2R2_4hEMA50_Trend_Filter"
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
    
    # Get daily data for Pivot Point calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Pivot Points for each day: based on prior day's OHLC
    # Pivot = (high + low + close) / 3
    # R1 = 2*Pivot - low
    # S1 = 2*Pivot - high
    # R2 = Pivot + (high - low)
    # S2 = Pivot - (high - low)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Align Pivot levels to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1.values)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2.values)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filters
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)  # Volume confirmation
    volume_expansion = volume > np.roll(volume, 1)  # Current volume > previous
    volume_expansion[0] = False
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMA50
        # Skip if any critical value is NaN or outside session
        if (np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or
            np.isnan(ema_50_4h[i]) or np.isnan(volume_filter[i]) or np.isnan(volume_expansion[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Fade at S1/R1: counter-trend entry when price reverses from these levels
            # Long fade at S1: price touches S1 and closes back above it
            if close[i] <= s1_4h[i] * 1.001 and close[i] > s1_4h[i] and volume_filter[i]:
                # Only take long fade if below 4h EMA50 (bearish context)
                if close[i] < ema_50_4h[i]:
                    signals[i] = 0.25
                    position = 1
            # Short fade at R1: price touches R1 and closes back below it
            elif close[i] >= r1_4h[i] * 0.999 and close[i] < r1_4h[i] and volume_filter[i]:
                # Only take short fade if above 4h EMA50 (bullish context)
                if close[i] > ema_50_4h[i]:
                    signals[i] = -0.25
                    position = -1
            # Breakout at S2/R2: trend-following entry when price breaks through
            # Long breakout at R2: price breaks above R2 with volume expansion
            elif close[i] > r2_4h[i] and volume_expansion[i]:
                # Only take long breakout if above 4h EMA50 (bullish context)
                if close[i] > ema_50_4h[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakout at S2: price breaks below S2 with volume expansion
            elif close[i] < s2_4h[i] and volume_expansion[i]:
                # Only take short breakout if below 4h EMA50 (bearish context)
                if close[i] < ema_50_4h[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price reaches R1 (profit target) or breaks below S2 (stop)
            if close[i] >= r1_4h[i] * 0.999:  # Take profit at R1
                signals[i] = 0.0
                position = 0
            elif close[i] < s2_4h[i]:  # Stop loss if breaks below S2
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches S1 (profit target) or breaks above R2 (stop)
            if close[i] <= s1_4h[i] * 1.001:  # Take profit at S1
                signals[i] = 0.0
                position = 0
            elif close[i] > r2_4h[i]:  # Stop loss if breaks above R2
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals