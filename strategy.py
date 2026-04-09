#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Uses 4h HTF for trend direction (EMA50 > EMA200 = uptrend, < = downtrend)
# - 1h Camarilla pivot levels (R3, S3, R4, S4) from prior 4h bar
# - Long on break above R4 in uptrend, short on break below S4 in downtrend
# - Volume confirmation: current 1h volume > 1.5x 20-period average
# - Session filter: trade only between 08:00-20:00 UTC
# - Fixed position size 0.20 to control drawdown and reduce fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Works in bull via breakouts in uptrend, works in bear via breakdowns in downtrend

name = "1h_4h_camarilla_breakout_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMAs for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 4h Camarilla pivot levels from prior bar
    typical_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    
    # Camarilla levels
    r3_4h = typical_4h + range_4h * 1.1 / 4
    s3_4h = typical_4h - range_4h * 1.1 / 4
    r4_4h = typical_4h + range_4h * 1.1 / 2
    s4_4h = typical_4h - range_4h * 1.1 / 2
    
    # Align all 4h data to 1h timeframe (wait for completed 4h bar)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, r4_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, s4_4h)
    
    # Pre-compute volume confirmation (20-period average for 1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Session filter: 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(ema_200_4h_aligned[i]) or np.isnan(r4_4h_aligned[i]) or
            np.isnan(s4_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: 4h EMA50 > EMA200 = uptrend, < = downtrend
        uptrend = ema_50_4h_aligned[i] > ema_200_4h_aligned[i]
        downtrend = ema_50_4h_aligned[i] < ema_200_4h_aligned[i]
        
        # Fixed position size
        position_size = 0.20
        
        if position == 1:  # Long position
            # Exit when price closes below 4h EMA50 (trend change)
            if close[i] < ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price closes above 4h EMA50 (trend change)
            if close[i] > ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Breakout entry with volume confirmation and trend alignment
            if volume_confirmed:
                # Long: break above R4 in uptrend
                if uptrend and close[i] > r4_4h_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: break below S4 in downtrend
                elif downtrend and close[i] < s4_4h_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals