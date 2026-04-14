#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation
# Long when price breaks above Donchian upper channel AND 1d EMA(50) rising AND volume > 1.5x average
# Short when price breaks below Donchian lower channel AND 1d EMA(50) falling AND volume > 1.5x average
# Exit when price crosses 10-period SMA OR Donchian middle line
# Donchian captures breakouts, 1d EMA filters higher timeframe trend, volume confirms institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

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
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate 10-period SMA for exit
    sma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean()
    
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
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(sma_10[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get EMA values aligned to 12h timeframe
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d.values)
        ema_val = ema_50_aligned[i]
        ema_prev = ema_50_aligned[i-1]
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: Price breaks above Donchian upper channel AND 1d EMA rising AND volume confirmation
            if (price > donch_high[i] and ema_val > ema_prev and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Price breaks below Donchian lower channel AND 1d EMA falling AND volume confirmation
            elif (price < donch_low[i] and ema_val < ema_prev and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price crosses below 10-period SMA OR Donchian middle line
            if (price < sma_10[i] or price < donch_mid[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price crosses above 10-period SMA OR Donchian middle line
            if (price > sma_10[i] or price > donch_mid[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_1dEMA_Volume"
timeframe = "12h"
leverage = 1.0