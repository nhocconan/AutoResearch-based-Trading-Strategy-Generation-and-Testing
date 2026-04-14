#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Long when Alligator jaws (SMA13) > teeth (SMA8) > lips (SMA5) AND 1d EMA(50) rising AND volume > 1.5x average
# Short when Alligator jaws < teeth < lips AND 1d EMA(50) falling AND volume > 1.5x average
# Exit when Alligator lines cross in opposite direction OR price crosses 8-period SMA
# Williams Alligator identifies trend alignment; 1d EMA filters higher timeframe trend; volume confirms institutional participation
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
    
    # Calculate Williams Alligator components (5, 8, 13 period SMAs)
    sma_5 = pd.Series(close).rolling(window=5, min_periods=5).mean()
    sma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean()
    sma_13 = pd.Series(close).rolling(window=13, min_periods=13).mean()
    
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
        if (np.isnan(sma_5[i]) or 
            np.isnan(sma_8[i]) or 
            np.isnan(sma_13[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get EMA values aligned to 4h timeframe
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d.values)
        ema_val = ema_50_aligned[i]
        ema_prev = ema_50_aligned[i-1]
        
        jaw = sma_13[i]   # Alligator jaws (13-period)
        teeth = sma_8[i]  # Alligator teeth (8-period)
        lips = sma_5[i]   # Alligator lips (5-period)
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: Alligator aligned bullish (jaws > teeth > lips) AND 1d EMA rising AND volume confirmation
            if (jaw > teeth > lips and ema_val > ema_prev and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Alligator aligned bearish (jaws < teeth < lips) AND 1d EMA falling AND volume confirmation
            elif (jaw < teeth < lips and ema_val < ema_prev and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator alignment breaks OR price crosses below 8-period SMA
            if (jaw <= teeth or price < sma_8[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator alignment breaks OR price crosses above 8-period SMA
            if (jaw >= teeth or price > sma_8[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsAlligator_1dEMA_Volume"
timeframe = "4h"
leverage = 1.0