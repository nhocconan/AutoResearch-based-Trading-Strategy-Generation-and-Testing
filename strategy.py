#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot Reversal with 1d Trend Filter
# Hypothesis: Camarilla pivot levels (R3/S3) act as strong reversal zones in 6h timeframe.
# In bull markets, price reverses upward from S3; in bear markets, reverses downward from R3.
# 1d trend filter (price vs 200 EMA) ensures trades align with higher timeframe direction.
# Volume confirmation filters out false breakouts. Target: 15-25 trades/year (60-100 over 4 years).

name = "6h_camarilla_pivot_reversal_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for pivot points
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = Close + ((High - Low) * 1.5000)
    # R3 = Close + ((High - Low) * 1.1250)
    # S3 = Close - ((High - Low) * 1.1250)
    # S4 = Close - ((High - Low) * 1.5000)
    daily_range = daily_high - daily_low
    r3 = daily_close + (daily_range * 1.125)
    s3 = daily_close - (daily_range * 1.125)
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    prev_r3 = np.roll(r3, 1)
    prev_s3 = np.roll(s3, 1)
    prev_r3[0] = prev_r3[1] if len(prev_r3) > 1 else 0
    prev_s3[0] = prev_s3[1] if len(prev_s3) > 1 else 0
    
    # Align to 6h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_daily, prev_r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, prev_s3)
    
    # 1d trend filter: price above/below 200 EMA
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_200[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below S3 or trend turns bearish
            if (low[i] < s3_aligned[i] or close[i] < ema_200[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above R3 or trend turns bullish
            if (high[i] > r3_aligned[i] or close[i] > ema_200[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price touches/bounces from S3 with volume and bullish trend
            if ((low[i] <= s3_aligned[i] or close[i] <= s3_aligned[i]) and 
                close[i] > ema_200[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches/rejects from R3 with volume and bearish trend
            elif ((high[i] >= r3_aligned[i] or close[i] >= r3_aligned[i]) and 
                  close[i] < ema_200[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals