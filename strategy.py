#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# Long when price breaks above Donchian(20) upper band AND 1d EMA(50) rising AND volume > 1.5x average
# Short when price breaks below Donchian(20) lower band AND 1d EMA(50) falling AND volume > 1.5x average
# Exit when price crosses Donchian(20) midline OR opposite signal
# Uses Donchian for price channel breakouts, EMA for trend filter, volume for confirmation
# Designed to capture breakouts in trending markets with institutional volume support
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_mid = (high_20 + low_20) / 2
    
    # Calculate EMA on 1d (50-period) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get EMA values aligned to 4h timeframe
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d.values)
        ema_val = ema_50_aligned[i]
        ema_prev = ema_50_aligned[i-1]
        
        donchian_upper = high_20[i]
        donchian_lower = low_20[i]
        donchian_mid_val = donchian_mid[i]
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: Price breaks above Donchian upper AND 1d EMA rising AND volume confirmation
            if (price > donchian_upper and ema_val > ema_prev and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Price breaks below Donchian lower AND 1d EMA falling AND volume confirmation
            elif (price < donchian_lower and ema_val < ema_prev and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price crosses below Donchian midline OR opposite signal
            if (price < donchian_mid_val or 
                (price < donchian_lower and ema_val < ema_prev and vol > vol_threshold)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price crosses above Donchian midline OR opposite signal
            if (price > donchian_mid_val or 
                (price > donchian_upper and ema_val > ema_prev and vol > vol_threshold)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_1dEMA_Volume"
timeframe = "4h"
leverage = 1.0