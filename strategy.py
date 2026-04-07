#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla Pivot with 1d Volume and Trend Filter
# Hypothesis: Camarilla pivot levels (S3/R3) act as strong support/resistance in ranging markets.
# We trade reversals at S3/R3 with volume confirmation and 1d trend filter to avoid counter-trend trades.
# Works in bull/bear by filtering trades with higher timeframe trend.
# Target: 20-40 trades/year (80-160 over 4 years).

name = "4h_camarilla_pivot_1d_volume_trend_v1"
timeframe = "4h"
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
    
    # Get daily data for pivot points and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (Camarilla formula)
    # Pivot = (H + L + C) / 3
    # S1 = C - (H - L) * 1.1/12, S2 = C - (H - L) * 1.1/6, S3 = C - (H - L) * 1.1/4
    # R1 = C + (H - L) * 1.1/12, R2 = C + (H - L) * 1.1/6, R3 = C + (H - L) * 1.1/4
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3
    range_hl = daily_high - daily_low
    s1 = daily_close - range_hl * 1.1 / 12
    s2 = daily_close - range_hl * 1.1 / 6
    s3 = daily_close - range_hl * 1.1 / 4
    r1 = daily_close + range_hl * 1.1 / 12
    r2 = daily_close + range_hl * 1.1 / 6
    r3 = daily_close + range_hl * 1.1 / 4
    
    # Align pivot levels to 4h timeframe
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    
    # 1d trend filter: 50-period EMA
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_4h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(s3_4h[i]) or np.isnan(r3_4h[i]) or 
            np.isnan(daily_ema_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R3 (take profit) or breaks below S3 (stop)
            if close[i] >= r3_4h[i] or close[i] <= s3_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches S3 (take profit) or breaks above R3 (stop)
            if close[i] <= s3_4h[i] or close[i] >= r3_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Only trade in direction of 1d trend to avoid counter-trend
                # Long: price touches or goes below S3 but closes back above it (in uptrend)
                if close[i] > s3_4h[i] and low[i] <= s3_4h[i] and close[i] > daily_ema_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches or goes above R3 but closes back below it (in downtrend)
                elif close[i] < r3_4h[i] and high[i] >= r3_4h[i] and close[i] < daily_ema_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals