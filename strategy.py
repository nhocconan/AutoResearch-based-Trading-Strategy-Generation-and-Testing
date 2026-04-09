#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Uses 1w HTF for trend direction (close > EMA50 = uptrend, < = downtrend)
# - 1d Donchian breakout: long on break above 20-period high, short on break below 20-period low
# - Volume confirmation: current 1d volume > 1.5x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 20-50 trades/year on 1d timeframe (80-200 total over 4 years)
# - Works in both bull and bear markets via trend-filtered breakouts

name = "1d_1w_donchian_breakout_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 1d timeframe (wait for completed 1w bar)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_ma_20[i]) or
            np.isnan(low_ma_20[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: 1w EMA50 > close = uptrend, < close = downtrend
        uptrend = ema_50_1w_aligned[i] > close[i]
        downtrend = ema_50_1w_aligned[i] < close[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when price closes below 1w EMA50 (trend change) or Donchian low
            if close[i] < ema_50_1w_aligned[i] or close[i] < low_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price closes above 1w EMA50 (trend change) or Donchian high
            if close[i] > ema_50_1w_aligned[i] or close[i] > high_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Breakout entry with volume confirmation and trend alignment
            if volume_confirmed:
                # Long: break above Donchian high in uptrend
                if uptrend and close[i] > high_ma_20[i]:
                    position = 1
                    signals[i] = position_size
                # Short: break below Donchian low in downtrend
                elif downtrend and close[i] < low_ma_20[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals