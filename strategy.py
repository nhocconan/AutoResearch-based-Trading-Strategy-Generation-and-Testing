#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter combined with 4-hour RSI extremes and volume confirmation.
# In ranging markets (CHOP > 61.8), use RSI mean reversion: long RSI < 30, short RSI > 70.
# In trending markets (CHOP < 38.2), use RSI momentum: long RSI > 50, short RSI < 50.
# Volume confirmation: current volume > 1.3 * 20-period EMA of volume.
# Designed for low trade frequency (target: 20-30/year) to minimize fee drag and improve generalization.
# Works in bull markets via trend-following RSI and in bear markets via mean-reversion RSI.
name = "4h_CHOP_RSI_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Choppiness Index (14-period)
    atr = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.3 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine market regime
            is_ranging = chop[i] > 61.8
            is_trending = chop[i] < 38.2
            
            if is_ranging:
                # RSI mean reversion in ranging market
                long_condition = rsi[i] < 30 and volume_spike[i]
                short_condition = rsi[i] > 70 and volume_spike[i]
            elif is_trending:
                # RSI momentum in trending market
                long_condition = rsi[i] > 50 and volume_spike[i]
                short_condition = rsi[i] < 50 and volume_spike[i]
            else:
                # Neutral chop: no trade
                long_condition = False
                short_condition = False
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI crosses 50 (mean reversion) or chop becomes too high
            if rsi[i] > 50 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI crosses 50 (mean reversion) or chop becomes too high
            if rsi[i] < 50 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals