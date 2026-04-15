#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) Breakout + Weekly Trend Filter + Volume Confirmation
# Long when price breaks above 20-day high AND weekly EMA50 is rising AND volume > 1.5x 20-day median volume
# Short when price breaks below 20-day low AND weekly EMA50 is falling AND volume > 1.5x 20-day median volume
# Exit when price crosses back through the opposite Donchian band or weekly trend reverses
# Designed for breakout trading with trend filtering to work in trending markets and avoid whipsaws
# Conservative sizing (0.25) to limit trade frequency (target: 20-50 trades/year)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly EMA slope (trend direction)
    ema_slope_1w = np.diff(ema_50_1w_aligned, prepend=ema_50_1w_aligned[0])
    
    # Volume confirmation: current > 1.5x median of last 20 days
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup for weekly EMA
        # Skip if any required data is NaN
        if (np.isnan(high_20.iloc[i]) or np.isnan(low_20.iloc[i]) or 
            np.isnan(ema_slope_1w[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above 20-day high, weekly EMA rising, volume spike
        if (close[i] > high_20.iloc[i] and 
            ema_slope_1w[i] > 0 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below 20-day low, weekly EMA falling, volume spike
        elif (close[i] < low_20.iloc[i] and 
              ema_slope_1w[i] < 0 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit conditions
        elif i > 0:
            prev_signal = signals[i-1]
            # Exit long: price crosses below 20-day low OR weekly trend turns down
            if (prev_signal == 0.25 and 
                (close[i] < low_20.iloc[i] or ema_slope_1w[i] <= 0)):
                signals[i] = 0.0
            # Exit short: price crosses above 20-day high OR weekly trend turns up
            elif (prev_signal == -0.25 and 
                  (close[i] > high_20.iloc[i] or ema_slope_1w[i] >= 0)):
                signals[i] = 0.0
            # Otherwise hold position
            else:
                signals[i] = prev_signal
    
    return signals

name = "1d_Donchian_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0