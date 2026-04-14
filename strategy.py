# 4h Volume Spike Breakout with 12h Trend Filter
# Strategy: Breakouts confirmed by volume spikes (>2x average) in direction of 12h EMA trend
# Exit when price crosses back through Donchian midpoint
# Works in bull/bear by following higher timeframe trend, volume filters out false breakouts
# Target: 20-40 trades/year to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Donchian channels on 4h (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate EMA on 12h (50-period) for trend filter
    ema_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    # Pre-calculate aligned EMA array (critical for performance)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50.values)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        ema_val = ema_50_aligned[i]
        ema_prev = ema_50_aligned[i-1]
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 2.0  # Require 2x volume spike for confirmation
        
        if position == 0:
            # Long setup: price breaks above Donchian upper AND 12h EMA rising AND volume spike
            if (high_val > highest_high[i] and ema_val > ema_prev and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below Donchian lower AND 12h EMA falling AND volume spike
            elif (low_val < lowest_low[i] and ema_val < ema_prev and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian middle OR opposite breakout with volume
            if (close_val < donchian_mid[i] or 
                (low_val < lowest_low[i] and vol > vol_threshold)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Donchian middle OR opposite breakout with volume
            if (close_val > donchian_mid[i] or 
                (high_val > highest_high[i] and vol > vol_threshold)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Volume_Spike_Breakout_12hTrend"
timeframe = "4h"
leverage = 1.0