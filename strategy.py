#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with 1-week trend filter and volume confirmation
# Long when Jaw > Teeth > Lips (bullish alignment) AND price > weekly EMA50 AND volume > 1.5x 20-period average
# Short when Jaw < Teeth < Lips (bearish alignment) AND price < weekly EMA50 AND volume > 1.5x 20-period average
# Exit when Alligator lines cross (alignment breaks) or price crosses weekly EMA50 in opposite direction
# Williams Alligator identifies trend phases; weekly EMA50 filters for higher timeframe trend; volume confirms strength
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Williams Alligator on 12h (Jaw=13, Teeth=8, Lips=5 SMAs with future shifts)
    close_series = pd.Series(close)
    # Jaw: 13-period SMA shifted 8 bars forward
    jaw = close_series.rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA shifted 5 bars forward
    teeth = close_series.rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA shifted 3 bars forward
    lips = close_series.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max shift 8 + buffer)
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Bullish alignment: Jaw > Teeth > Lips
            bullish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
            # Bearish alignment: Jaw < Teeth < Lips
            bearish_alignment = jaw[i] < teeth[i] and teeth[i] < lips[i]
            
            # Long setup: bullish alignment + price > weekly EMA50 + volume confirmation
            if (bullish_alignment and price > ema50_1w_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: bearish alignment + price < weekly EMA50 + volume confirmation
            elif (bearish_alignment and price < ema50_1w_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: alignment breaks or price crosses below weekly EMA50
            bullish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
            if not bullish_alignment or price < ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: alignment breaks or price crosses above weekly EMA50
            bearish_alignment = jaw[i] < teeth[i] and teeth[i] < lips[i]
            if not bearish_alignment or price > ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_WilliamsAlligator_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0