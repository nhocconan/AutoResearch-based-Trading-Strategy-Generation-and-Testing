#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation
# - Uses 12h HTF for trend direction (EMA50 > EMA200 = uptrend, < = downtrend)
# - 4h Donchian channel (20-period high/low) for breakout entries
# - Long on break above upper band in uptrend, short on break below lower band in downtrend
# - Volume confirmation: current 4h volume > 1.5x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_12h_donchian_breakout_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMAs for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMAs to 4h timeframe (wait for completed 12h bar)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Pre-compute 4h Donchian channel (20-period)
    donchian_window = 20
    upper_band = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_band = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_200_12h_aligned[i]) or
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: 12h EMA50 > EMA200 = uptrend, < = downtrend
        uptrend = ema_50_12h_aligned[i] > ema_200_12h_aligned[i]
        downtrend = ema_50_12h_aligned[i] < ema_200_12h_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when price closes below 12h EMA50 (trend change) or Donchian lower band
            if close[i] < ema_50_12h_aligned[i] or close[i] < lower_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price closes above 12h EMA50 (trend change) or Donchian upper band
            if close[i] > ema_50_12h_aligned[i] or close[i] > upper_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Breakout entry with volume confirmation and trend alignment
            if volume_confirmed:
                # Long: break above upper band in uptrend
                if uptrend and close[i] > upper_band[i]:
                    position = 1
                    signals[i] = position_size
                # Short: break below lower band in downtrend
                elif downtrend and close[i] < lower_band[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals