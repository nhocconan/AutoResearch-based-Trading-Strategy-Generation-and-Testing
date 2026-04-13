#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index (CHOP) regime filter + Donchian(20) breakout + volume confirmation.
# CHOP > 61.8 indicates ranging market (mean reversion), CHOP < 38.2 indicates trending (breakout).
# In trending regimes (CHOP < 38.2), we take Donchian breakouts in direction of trend (price > SMA50).
# In ranging regimes (CHOP > 61.8), we fade Donchian breakouts (mean reversion at extremes).
# Volume confirmation ensures breakouts have institutional participation.
# Works in both bull and bear markets by adapting to regime.
# Target: 20-50 trades per year (80-200 total over 4 years) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate True Range for CHOP
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range: max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value has no previous close
    
    # ATR(14) for CHOP denominator
    atr_period = 14
    atr = np.zeros(len(tr))
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate highest high and lowest low over 14 days for CHOP numerator
    highest_high = np.zeros(len(high_1d))
    lowest_low = np.zeros(len(low_1d))
    for i in range(14, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-14:i+1])
        lowest_low[i] = np.min(low_1d[i-14:i+1])
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14)/ (HHV - LLV)) / log10(14)
    chop = np.zeros(len(high_1d))
    for i in range(14, len(high_1d)):
        if highest_high[i] > lowest_low[i]:
            sum_atr = np.sum(atr[i-14:i+1])
            chop[i] = 100 * np.log10(sum_atr / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # Neutral when no range
    
    # Chop values before period remain NaN
    chop[:14] = np.nan
    
    # Align CHOP to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h indicators
    # Donchian channels (20-period)
    highest_high_20 = np.full(n, np.nan)
    lowest_low_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_high_20[i] = np.max(high[i-20:i+1])
        lowest_low_20[i] = np.min(low[i-20:i+1])
    
    # SMA(50) for trend filter
    close_s = pd.Series(close)
    sma50 = close_s.rolling(window=50, min_periods=50).mean().values
    
    # Average volume (20-period) for confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(sma50[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        chop_val = chop_aligned[i]
        upper_channel = highest_high_20[i]
        lower_channel = lowest_low_20[i]
        sma50_val = sma50[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Trending regime: CHOP < 38.2 - trade breakouts in trend direction
            if chop_val < 38.2:
                # Long breakout: price above upper channel + above SMA50 + volume
                if (price > upper_channel and 
                    price > sma50_val and 
                    volume_confirm):
                    position = 1
                    signals[i] = position_size
                # Short breakout: price below lower channel + below SMA50 + volume
                elif (price < lower_channel and 
                      price < sma50_val and 
                      volume_confirm):
                    position = -1
                    signals[i] = -position_size
            # Ranging regime: CHOP > 61.8 - fade extremes (mean reversion)
            elif chop_val > 61.8:
                # Long at lower channel: price at support + below SMA50 (fade downtrend) + volume
                if (abs(price - lower_channel) < 0.001 * price and  # Near lower channel
                    price < sma50_val and 
                    volume_confirm):
                    position = 1
                    signals[i] = position_size
                # Short at upper channel: price at resistance + above SMA50 (fade uptrend) + volume
                elif (abs(price - upper_channel) < 0.001 * price and  # Near upper channel
                      price > sma50_val and 
                      volume_confirm):
                    position = -1
                    signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below SMA50 or reaches opposite channel
            if (price < sma50_val or 
                price < lower_channel):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above SMA50 or reaches opposite channel
            if (price > sma50_val or 
                price > upper_channel):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_CHOP_Donchian_MeanRev_Trend"
timeframe = "4h"
leverage = 1.0