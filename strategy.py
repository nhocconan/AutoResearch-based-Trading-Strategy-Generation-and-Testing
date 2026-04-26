#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Reversal_1dTrendFilter_v1
Hypothesis: 6h Camarilla R3/S3 reversal strategy with daily trend filter.
- Uses 6h timeframe for moderate trade frequency (target: 50-150 total trades over 4 years)
- Camarilla pivot levels calculated from daily OHLC (R3, S3, R4, S4)
- Long when price crosses below S3 with bullish daily trend (close > EMA50)
- Short when price crosses above R3 with bearish daily trend (close < EMA50)
- Uses volume confirmation (volume > 1.5x 20-period average) to avoid false breakouts
- Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by fading extreme intraday moves against the daily trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily OHLC for Camarilla pivots
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla pivot calculations (based on previous day's range)
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    daily_range = daily_high - daily_low
    r3 = daily_close + (daily_range * 1.1 / 4)
    s3 = daily_close - (daily_range * 1.1 / 4)
    r4 = daily_close + (daily_range * 1.1 / 2)
    s4 = daily_close - (daily_range * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (wait for daily bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA and 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filters
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Price levels
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        r4_level = r4_aligned[i]
        s4_level = s4_aligned[i]
        
        if position == 0:
            # Long: price crosses below S3 with volume AND daily uptrend
            if (close[i] <= s3_level and close[i-1] > s3_level) and vol_confirm and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price crosses above R3 with volume AND daily downtrend
            elif (close[i] >= r3_level and close[i-1] < r3_level) and vol_confirm and daily_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price crosses above R3 (reversal) OR hits R4 (stop)
            if close[i] >= r3_level or close[i] >= r4_level:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price crosses below S3 (reversal) OR hits S4 (stop)
            if close[i] <= s3_level or close[i] <= s4_level:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Reversal_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0