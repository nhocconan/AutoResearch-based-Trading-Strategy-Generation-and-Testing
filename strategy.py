#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels with breakout and fade logic
# - Fade (counter-trend) at R3/S3 levels when price reverses from these levels with volume confirmation
# - Breakout (trend-following) at R4/S4 levels when price breaks through with volume expansion
# - Uses 12h Camarilla levels calculated from prior 12h bar's range
# - Adds 1d EMA34 filter to only take trades in direction of daily trend
# - Designed to work in ranging markets (fades at R3/S3) and trending markets (breakouts at R4/S4)
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_Camarilla_R3S3_R4S4_1dEMA34_Trend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 12h bar: based on prior bar's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    prev_close = df_12h['close'].shift(1)
    prev_high = df_12h['high'].shift(1)
    prev_low = df_12h['low'].shift(1)
    
    # Avoid division by zero in case of doji bars
    price_range = prev_high - prev_low
    price_range = np.where(price_range == 0, 0.0001, price_range)
    
    R4 = prev_close + 1.5 * price_range
    R3 = prev_close + 1.1 * price_range
    S3 = prev_close - 1.1 * price_range
    S4 = prev_close - 1.5 * price_range
    
    # Align Camarilla levels to 6h timeframe
    R4_6h = align_htf_to_ltf(prices, df_12h, R4.values)
    R3_6h = align_htf_to_ltf(prices, df_12h, R3.values)
    S3_6h = align_htf_to_ltf(prices, df_12h, S3.values)
    S4_6h = align_htf_to_ltf(prices, df_12h, S4.values)
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
    
    for i in range(20, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(R4_6h[i]) or np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or np.isnan(S4_6h[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_filter[i]) or np.isnan(volume_expansion[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Fade at R3/S3: counter-trend entry when price reverses from these levels
            # Long fade at S3: price touches S3 and closes back above it
            if close[i] <= S3_6h[i] * 1.001 and close[i] > S3_6h[i] and volume_filter[i]:
                # Only take long fade if below daily EMA (bearish context)
                if close[i] < ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short fade at R3: price touches R3 and closes back below it
            elif close[i] >= R3_6h[i] * 0.999 and close[i] < R3_6h[i] and volume_filter[i]:
                # Only take short fade if above daily EMA (bullish context)
                if close[i] > ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            # Breakout at R4/S4: trend-following entry when price breaks through
            # Long breakout at R4: price breaks above R4 with volume expansion
            elif close[i] > R4_6h[i] and volume_expansion[i]:
                # Only take long breakout if above daily EMA (bullish context)
                if close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakout at S4: price breaks below S4 with volume expansion
            elif close[i] < S4_6h[i] and volume_expansion[i]:
                # Only take short breakout if below daily EMA (bearish context)
                if close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price reaches R3 (profit target) or breaks below S4 (stop)
            if close[i] >= R3_6h[i] * 0.999:  # Take profit at R3
                signals[i] = 0.0
                position = 0
            elif close[i] < S4_6h[i]:  # Stop loss if breaks below S4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches S3 (profit target) or breaks above R4 (stop)
            if close[i] <= S3_6h[i] * 1.001:  # Take profit at S3
                signals[i] = 0.0
                position = 0
            elif close[i] > R4_6h[i]:  # Stop loss if breaks above R4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals