#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# - Uses 1d HTF for trend direction (EMA50 > EMA200 = uptrend, < = downtrend)
# - 12h Donchian channel (20-period) breakout for entries
# - Long on break above upper band in uptrend, short on break below lower band in downtrend
# - Volume confirmation: current 12h volume > 1.5x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 12-25 trades/year on 12h timeframe (48-100 total over 4 years)
# - Works in bull markets via trend-following breakouts
# - Works in bear markets via short breakdowns in downtrends

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
    
    # Calculate 1d EMAs for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMAs to 12h timeframe (wait for completed 1d bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Pre-compute 12h Donchian channel (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: 1d EMA50 > EMA200 = uptrend, < = downtrend
        uptrend = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
        downtrend = ema_50_1d_aligned[i] < ema_200_1d_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when price closes below 12h 20-period low (Donchian lower band)
            if close[i] < low_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price closes above 12h 20-period high (Donchian upper band)
            if close[i] > high_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Breakout entry with volume confirmation and trend alignment
            if volume_confirmed:
                # Long: break above upper Donchian band in uptrend
                if uptrend and close[i] > high_ma_20[i]:
                    position = 1
                    signals[i] = position_size
                # Short: break below lower Donchian band in downtrend
                elif downtrend and close[i] < low_ma_20[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals