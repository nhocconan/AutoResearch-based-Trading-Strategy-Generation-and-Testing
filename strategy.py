#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA(34) trend filter and volume confirmation
# Long when Alligator jaws < teeth < lips (bullish alignment), price > 1d EMA34, and volume > 2.0x 20-bar average
# Short when Alligator jaws > teeth > lips (bearish alignment), price < 1d EMA34, and volume > 2.0x 20-bar average
# Williams Alligator uses SMAs of median price (HLC/3) with specific periods: jaws=13, teeth=8, lips=5
# The Alligator indicator identifies trending vs ranging markets - ideal for 6h timeframe
# Volume confirmation ensures breakout strength
# Discrete position sizing (0.25) to minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Works in bull (bullish alignment + rising EMA) and bear (bearish alignment + falling EMA)

name = "6h_WilliamsAlligator_1dEMA34_Volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 6h
    # Median price = (high + low + close) / 3
    median_price = (high + low + close) / 3.0
    
    # Alligator lines: SMAs of median price with specific periods
    # Jaws (Blue): 13-period SMA, shifted 8 bars
    # Teeth (Red): 8-period SMA, shifted 5 bars  
    # Lips (Green): 5-period SMA, shifted 3 bars
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    # Williams Alligator needs max(13,8,5) + max shift(8,5,3) = 13+8 = 21, plus EMA(34) and volume MA(20)
    start_idx = max(34, 20, 21) + 1
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaws[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: jaws < teeth < lips
            bullish_alignment = (jaws[i] < teeth[i]) and (teeth[i] < lips[i])
            # Bearish Alligator alignment: jaws > teeth > lips
            bearish_alignment = (jaws[i] > teeth[i]) and (teeth[i] > lips[i])
            
            # Long entry: bullish alignment, price > 1d EMA34, volume spike
            if (bullish_alignment and 
                close[i] > ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment, price < 1d EMA34, volume spike
            elif (bearish_alignment and 
                  close[i] < ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: bearish Alligator alignment or price < 1d EMA34
            bearish_alignment = (jaws[i] > teeth[i]) and (teeth[i] > lips[i])
            if (bearish_alignment or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bullish Alligator alignment or price > 1d EMA34
            bullish_alignment = (jaws[i] < teeth[i]) and (teeth[i] < lips[i])
            if (bullish_alignment or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals