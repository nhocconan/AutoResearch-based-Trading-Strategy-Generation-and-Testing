#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band width regime filter + Bollinger Band breakout with volume confirmation
# Uses Bollinger Band width percentile to identify low volatility (squeeze) conditions
# Then enters on breakout of Bollinger Bands with volume confirmation
# Works in both bull and bear markets by trading volatility breakouts
# Target: 20-30 trades/year per symbol
name = "12h_BB_Width_Squeeze_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    
    # Calculate Bollinger Bands
    sma = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    std = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    upper_band = sma + bb_mult * std
    lower_band = sma - bb_mult * std
    
    # Bollinger Band Width
    bb_width = (upper_band - lower_band) / sma
    
    # Bollinger Band Width Percentile (50-period lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).apply(
        lambda x: np.percentile(x, 50) if len(x) == 50 else np.nan, raw=True
    ).values
    
    # Squeeze condition: BB width below 50th percentile (low volatility)
    squeeze_condition = bb_width < bb_width_percentile
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_length, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(squeeze_condition[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.8 * vol_ma
        
        # Bollinger Band breakout conditions
        breakout_up = price > upper_band[i]
        breakout_down = price < lower_band[i]
        
        if position == 0:
            # Enter long on upward breakout during squeeze with volume confirmation
            if breakout_up and squeeze_condition[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short on downward breakout during squeeze with volume confirmation
            elif breakout_down and squeeze_condition[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to middle band (mean reversion) or volatility expands
            if price < sma[i] or not squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to middle band or volatility expands
            if price > sma[i] or not squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals