#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator: Jaw (EMA13, 8-bar shift), Teeth (EMA8, 5-bar shift), Lips (EMA5, 3-bar shift)
# Long when Lips > Teeth > Jaw (bullish alignment) AND close > 1d EMA50 AND volume > 1.8x 20-bar average
# Short when Lips < Teeth < Jaw (bearish alignment) AND close < 1d EMA50 AND volume > 1.8x 20-bar average
# Uses Williams Alligator for trend identification, EMA50 for higher-timeframe trend filter, volume for momentum confirmation
# Designed for low trade frequency (~12-37/year on 12h) to minimize fee drag
# Works in bull (trend continuation with rising volume) and bear (trend continuation with rising volume)

name = "12h_WilliamsAlligator_EMA50_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator components on 12h data
    # Jaw: Blue line - EMA(13) of median price, shifted 8 bars
    # Teeth: Red line - EMA(8) of median price, shifted 5 bars  
    # Lips: Green line - EMA(5) of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    jaw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (1.8x 20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 20) + 8  # EMA50(1d) + volume MA(20) + Alligator jaw shift(8)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Lips > Teeth > Jaw (bullish alignment), close > 1d EMA50, volume spike
            if (lips[i] > teeth[i] > jaw[i] and 
                close[i] > ema_50_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Lips < Teeth < Jaw (bearish alignment), close < 1d EMA50, volume spike
            elif (lips[i] < teeth[i] < jaw[i] and 
                  close[i] < ema_50_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: bearish alignment (Lips < Teeth < Jaw) or close < 1d EMA50 (trend failure)
            if (lips[i] < teeth[i] < jaw[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bullish alignment (Lips > Teeth > Jaw) or close > 1d EMA50 (trend failure)
            if (lips[i] > teeth[i] > jaw[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals