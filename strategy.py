#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Uses 1w HTF for trend direction (EMA20 > EMA50 = uptrend, < = downtrend)
# - 12h Camarilla pivot levels (R3, S3, R4, S4) from prior 1d bar
# - Long on break above R4 in uptrend, short on break below S4 in downtrend
# - Volume confirmation: current 12h volume > 1.4x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_camarilla_breakout_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla pivot levels from prior bar
    typical_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3_1d = typical_1d + range_1d * 1.1 / 4
    s3_1d = typical_1d - range_1d * 1.1 / 4
    r4_1d = typical_1d + range_1d * 1.1 / 2
    s4_1d = typical_1d - range_1d * 1.1 / 2
    
    # Calculate 1w EMAs for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF data to 12h timeframe (wait for completed HTF bar)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.4x average
        volume_confirmed = volume[i] > 1.4 * vol_ma_20[i]
        
        # Trend filter: 1w EMA20 > EMA50 = uptrend, < = downtrend
        uptrend = ema_20_1w_aligned[i] > ema_50_1w_aligned[i]
        downtrend = ema_20_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when price closes below 1d EMA20 (trend change)
            ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
            ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
            if close[i] < ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price closes above 1d EMA20 (trend change)
            ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
            ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
            if close[i] > ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Breakout entry with volume confirmation and trend alignment
            if volume_confirmed:
                # Long: break above R4 in uptrend
                if uptrend and close[i] > r4_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: break below S4 in downtrend
                elif downtrend and close[i] < s4_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals