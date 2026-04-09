#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# - Uses 1d HTF for trend direction (EMA50 > EMA200 = uptrend, < = downtrend)
# - 12h Donchian channels (20-period) for breakout entries
# - Long on break above upper channel in uptrend, short on break below lower channel in downtrend
# - Volume confirmation: current 12h volume > 1.4x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 12-30 trades/year on 12h timeframe (48-120 total over 4 years)

name = "12h_1d_donchian_breakout_trend_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_12h = df_1d['high'].values  # Reuse for Donchian calculation (will be aligned)
    low_12h = df_1d['low'].values
    
    # Calculate 1d EMAs for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d Donchian channels (20-period)
    upper_20_1d = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_20_1d = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align all 1d data to 12h timeframe (wait for completed 1d bar)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    upper_20_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_20_1d)
    lower_20_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_20_1d)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(upper_20_1d_aligned[i]) or
            np.isnan(lower_20_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.4x average
        volume_confirmed = volume[i] > 1.4 * vol_ma_20[i]
        
        # Trend filter: 1d EMA50 > EMA200 = uptrend, < = downtrend
        uptrend = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
        downtrend = ema_50_1d_aligned[i] < ema_200_1d_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when price closes below 1d EMA50 (trend change)
            if close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price closes above 1d EMA50 (trend change)
            if close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Breakout entry with volume confirmation and trend alignment
            if volume_confirmed:
                # Long: break above upper Donchian in uptrend
                if uptrend and close[i] > upper_20_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: break below lower Donchian in downtrend
                elif downtrend and close[i] < lower_20_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals