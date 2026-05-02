#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w EMA34 trend filter + volume confirmation
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) to identify trend direction and strength
# Weekly EMA34 ensures alignment with higher timeframe bias for both bull and bear markets
# Volume confirmation (1.5x 20-period average) filters for institutional participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 30-80 total trades over 4 years (7-20/year) for 1d timeframe
# Alligator provides clear trend signals while weekly filter prevents counter-trend trading

name = "1d_WilliamsAlligator_1wEMA34_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: Jaw(13), Teeth(8), Lips(5) - all SMAs of median price
    median_price = (high + low) / 2
    median_series = pd.Series(median_price)
    
    jaw = median_series.rolling(window=13, min_periods=13).mean().shift(1).values  # Jaw (13)
    teeth = median_series.rolling(window=8, min_periods=8).mean().shift(1).values   # Teeth (8)
    lips = median_series.rolling(window=5, min_periods=5).mean().shift(1).values    # Lips (5)
    
    # Weekly trend filter: price > weekly EMA34 for longs, < for shorts
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator bullish: Lips > Teeth > Jaw (green alignment)
            # Alligator bearish: Lips < Teeth < Jaw (red alignment)
            if lips[i] > teeth[i] and teeth[i] > jaw[i]:  # Bullish alignment
                if volume_spike[i] and close[i] > ema34_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif lips[i] < teeth[i] and teeth[i] < jaw[i]:  # Bearish alignment
                if volume_spike[i] and close[i] < ema34_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish or price < weekly EMA34
            if lips[i] < teeth[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish or price > weekly EMA34
            if lips[i] > teeth[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals