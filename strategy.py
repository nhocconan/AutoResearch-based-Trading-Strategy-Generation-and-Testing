#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation
# Designed to work in bull markets (trend-following breakouts) and bear markets (mean-reversion at band edges)
# Uses discrete position sizing (0.25) to limit overtrading and fee drag
# Volume confirmation ensures institutional participation
# Target: 20-40 trades/year to avoid fee drag while capturing significant moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour EMA for trend filter (calculated once)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 4-hour Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max()
    lower_channel = low_series.rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: current > 1.5x average of last 20 bars
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=1).mean()
    vol_threshold = 1.5 * vol_avg
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above upper channel + above 12h EMA + volume confirmation
        if (close[i] > upper_channel[i] and 
            close[i] > ema_12h_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below lower channel + below 12h EMA + volume confirmation
        elif (close[i] < lower_channel[i] and 
              close[i] < ema_12h_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle of channel (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (upper_channel[i] + lower_channel[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (upper_channel[i] + lower_channel[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Breakout_EMA50_Volume"
timeframe = "4h"
leverage = 1.0