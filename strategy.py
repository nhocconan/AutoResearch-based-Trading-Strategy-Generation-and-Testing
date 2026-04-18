#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter + 1w EMA200 trend filter + Volume confirmation
# Choppiness Index identifies trending vs ranging markets (CHOP > 61.8 = range, CHOP < 38.2 = trend).
# In trending markets (CHOP < 38.2): trade in direction of weekly EMA200.
# In ranging markets (CHOP > 61.8): mean-revert at daily Bollinger Bands (20,2).
# Volume confirmation filters low-quality signals.
# Works in bull markets (trend following) and bear markets (mean reversion in ranges).
# Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
name = "1d_Choppiness_EMA200_Bollinger_Volume"
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
    
    # Get 1d data (same as primary for indicators) and 1w for EMA200
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index (14-period) on 1d data
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(n)
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.absolute(np.roll(close, 1) - low)))
    tr1[0] = high[0] - low[0]  # First TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr1 * 14 / (max_high - min_low)) / np.log10(14)
    
    # Calculate EMA200 on 1w data for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Bollinger Bands (20,2) on 1d data for mean reversion
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop[i]) or np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        chop_val = chop[i]
        ema_val = ema_200_1w_aligned[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        
        if position == 0:
            # Trending market (CHOP < 38.2): follow weekly EMA200 trend
            if chop_val < 38.2:
                if close_val > ema_val and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                elif close_val < ema_val and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market (CHOP > 61.8): mean revert at Bollinger Bands
            elif chop_val > 61.8:
                if close_val <= bb_lower[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                elif close_val >= bb_upper[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: 
            # - Trending: close below weekly EMA200
            # - Ranging: touch upper Bollinger Band
            # - Opposite regime extreme
            if ((chop_val < 38.2 and close_val < ema_val) or
                (chop_val > 61.8 and close_val >= bb_upper_val) or
                (chop_val >= 38.2 and chop_val <= 61.8)):  # Transition zone
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit:
            # - Trending: close above weekly EMA200
            # - Ranging: touch lower Bollinger Band
            # - Opposite regime extreme
            if ((chop_val < 38.2 and close_val > ema_val) or
                (chop_val > 61.8 and close_val <= bb_lower_val) or
                (chop_val >= 38.2 and chop_val <= 61.8)):  # Transition zone
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals