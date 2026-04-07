#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Camarilla Pivot + Volume + Trend Filter
# Hypothesis: Price breaking above weekly R4 or below S4 indicates strong momentum
# continuation, while retracements to R3/S3 offer mean-reversion opportunities.
# Volume confirms institutional participation. Trend filter (price vs 50 EMA) ensures
# alignment with intermediate trend. Works in bull/bear markets: in bull, favor long
# breakouts/retracements; in bear, favor short breakdowns/retracements.
# Target: 15-35 trades/year (60-140 over 4 years).

name = "6h_weekly_camarilla_pivot_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (based on previous week)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high[0] = prev_weekly_high[1] if len(prev_weekly_high) > 1 else 0
    prev_weekly_low[0] = prev_weekly_low[1] if len(prev_weekly_low) > 1 else 0
    prev_weekly_close[0] = prev_weekly_close[1] if len(prev_weekly_close) > 1 else 0
    
    # Calculate Camarilla levels for previous week
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    weekly_range = prev_weekly_high - prev_weekly_low
    r4 = prev_weekly_close + 1.5 * weekly_range
    r3 = prev_weekly_close + 1.1 * weekly_range
    s3 = prev_weekly_close - 1.1 * weekly_range
    s4 = prev_weekly_close - 1.5 * weekly_range
    
    # Align weekly Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    
    # 50 EMA trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume filter: volume > 1.8x 24-period average (4 days for 6h)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below S3 or trend turns bearish or volume drops
            if (low[i] < s3_aligned[i] or close[i] < ema_50[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above R3 or trend turns bullish or volume drops
            if (high[i] > r3_aligned[i] or close[i] > ema_50[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Breakout entries: strong momentum continuation
            # Long breakout: price breaks above weekly R4 with volume and bullish trend
            if ((high[i] > r4_aligned[i] or close[i] > r4_aligned[i]) and 
                close[i] > ema_50[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below weekly S4 with volume and bearish trend
            elif ((low[i] < s4_aligned[i] or close[i] < s4_aligned[i]) and 
                  close[i] < ema_50[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
            # Mean-reversion entries: retracement to support/resistance
            # Long retracement: price approaches weekly S3 with bullish bias and volume
            elif ((low[i] <= s3_aligned[i] * 1.002 or close[i] <= s3_aligned[i] * 1.002) and 
                  close[i] > ema_50[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.20  # Smaller size for mean reversion
            # Short retracement: price approaches weekly R3 with bearish bias and volume
            elif ((high[i] >= r3_aligned[i] * 0.998 or close[i] >= r3_aligned[i] * 0.998) and 
                  close[i] < ema_50[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.20  # Smaller size for mean reversion
    
    return signals