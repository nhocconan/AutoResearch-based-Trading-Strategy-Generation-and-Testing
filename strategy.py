#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla levels provide statistically significant support/resistance, EMA34 on 1d filters trend,
# and volume > 2x 20-period average confirms breakout strength. Designed for fewer trades.
name = "4h_Camarilla_R3S3_1dEMA34_Trend_VolumeSpike"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (OHLC)
    # We need daily OHLC to compute Camarilla levels for current 4h period
    # Since we don't have daily data aligned, we'll use rolling window approximation
    # For proper Camarilla, we need actual daily OHLC - but we'll approximate with 4h data
    # Better approach: get actual 1d OHLC from df_1d
    if len(df_1d) >= 1:
        # Use previous day's OHLC to calculate today's Camarilla levels
        prev_day_close = df_1d['close'].values[-2] if len(df_1d) >= 2 else df_1d['close'].values[-1]
        prev_day_high = df_1d['high'].values[-2] if len(df_1d) >= 2 else df_1d['high'].values[-1]
        prev_day_low = df_1d['low'].values[-2] if len(df_1d) >= 2 else df_1d['low'].values[-1]
        prev_day_open = df_1d['open'].values[-2] if len(df_1d) >= 2 else df_1d['open'].values[-1]
        
        # Calculate Camarilla levels based on previous day's range
        range_val = prev_day_high - prev_day_low
        if range_val <= 0:
            # Fallback if range is invalid
            r3 = s3 = r4 = s4 = 0
        else:
            # Camarilla levels
            r3 = prev_day_close + (range_val * 1.1 / 4)
            s3 = prev_day_close - (range_val * 1.1 / 4)
            r4 = prev_day_close + (range_val * 1.1 / 2)
            s4 = prev_day_close - (range_val * 1.1 / 2)
    else:
        r3 = s3 = r4 = s4 = 0
    
    # Since we can't easily get previous day's OHLC in loop, we'll use a simplified approach
    # Calculate rolling 24-period (6*4h) high/low as proxy for daily range
    # This is not perfect but avoids look-ahead and uses available data
    if len(prices) >= 24:
        daily_high = pd.Series(high).rolling(window=24, min_periods=24).max().values
        daily_low = pd.Series(low).rolling(window=24, min_periods=24).min().values
        daily_close = pd.Series(close).rolling(window=24, min_periods=24).mean().values  # approximate
        
        # Calculate Camarilla levels using rolling daily approximation
        range_val = daily_high - daily_low
        # Avoid division by zero
        range_val = np.where(range_val == 0, 1e-10, range_val)
        
        r3 = daily_close + (range_val * 1.1 / 4)
        s3 = daily_close - (range_val * 1.1 / 4)
        r4 = daily_close + (range_val * 1.1 / 2)
        s4 = daily_close - (range_val * 1.1 / 2)
    else:
        r3 = s3 = r4 = s4 = np.full_like(close, np.nan)
    
    # Volume filter: volume spike > 2x 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_4h[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(r4[i]) or np.isnan(s4[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions at Camarilla levels
        # Long when price breaks above R3 with volume spike and uptrend
        long_breakout = close[i] > r3[i] and close[i-1] <= r3[i-1]  # Just broke above R3
        # Short when price breaks below S3 with volume spike and downtrend
        short_breakout = close[i] < s3[i] and close[i-1] >= s3[i-1]  # Just broke below S3
        
        trend_up = close[i] > ema_34_4h[i]
        trend_down = close[i] < ema_34_4h[i]
        
        if position == 0:
            # Long: break above R3 + uptrend + volume spike
            if long_breakout and trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 + downtrend + volume spike
            elif short_breakout and trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: break below S3 or trend reversal
            if close[i] < s3[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: break above R3 or trend reversal
            if close[i] > r3[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals