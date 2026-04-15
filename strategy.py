#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w trend filter + volume confirmation
# Long when price breaks above 20-day high AND weekly EMA20 is rising AND volume > 1.5x 20-day median
# Short when price breaks below 20-day low AND weekly EMA20 is falling AND volume > 1.5x 20-day median
# Exit when price returns to the 10-day EMA or opposite breakout occurs
# Designed to capture strong trending moves with volatility expansion, avoiding choppy markets
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 10-day EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean()
    
    # 1-week EMA20 for trend filter (using weekly data)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean()
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w.values)
    
    # Weekly EMA slope (trend direction)
    ema_slope_1w = np.diff(ema_20_1w_aligned, prepend=ema_20_1w_aligned[0])
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup for Donchian
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_slope_1w[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above 20-day high, weekly EMA rising, volume spike
        if (close[i] > high_20[i] and 
            ema_slope_1w[i] > 0 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below 20-day low, weekly EMA falling, volume spike
        elif (close[i] < low_20[i] and 
              ema_slope_1w[i] < 0 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price returns to 10-day EMA or opposite breakout with volume
        elif i > 0 and signals[i-1] != 0:
            prev_signal = signals[i-1]
            exit_condition = False
            
            if prev_signal == 0.25:  # Long position
                # Exit if price crosses below 10-day EMA
                if close[i] < ema_10[i]:
                    exit_condition = True
                # Or if price breaks below 20-day low with volume (contrarian signal)
                elif (close[i] < low_20[i] and 
                      volume[i] > vol_threshold[i]):
                    exit_condition = True
                    
            elif prev_signal == -0.25:  # Short position
                # Exit if price crosses above 10-day EMA
                if close[i] > ema_10[i]:
                    exit_condition = True
                # Or if price breaks above 20-day high with volume (contrarian signal)
                elif (close[i] > high_20[i] and 
                      volume[i] > vol_threshold[i]):
                    exit_condition = True
            
            if exit_condition:
                signals[i] = 0.0
            else:
                signals[i] = prev_signal  # Hold position
        else:
            signals[i] = signals[i-1] if i > 0 else 0.0  # Hold or flat
    
    return signals

name = "1d_Donchian20_WeeklyEMA20_Volume"
timeframe = "1d"
leverage = 1.0