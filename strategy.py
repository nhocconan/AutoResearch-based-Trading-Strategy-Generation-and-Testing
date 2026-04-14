#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter with Donchian breakout and volume confirmation
# In high chop (range): mean-reversion at Donchian bands (sell at upper, buy at lower)
# In low chop (trend): trend-following breakouts (buy breakout, sell breakdown)
# Volume > 1.5x average confirms breakout strength
# This adapts to market regime, reducing false breakouts in ranges and capturing trends
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels on 4h (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index on daily (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    chop = 100 * np.log10(atr14 / range_14) / np.log10(14)
    chop[np.isnan(chop)] = 50  # neutral when undefined
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Donchian needs 20, chop needs extra
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Determine regime: chop > 61.8 = range (mean revert), chop < 38.2 = trend (trend follow)
            if chop_val > 61.8:
                # Range regime: mean reversion at Donchian bands
                if price <= low_20[i] and vol > vol_threshold:  # near lower band -> buy
                    position = 1
                    signals[i] = position_size
                elif price >= high_20[i] and vol > vol_threshold:  # near upper band -> sell
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                # Trend regime: breakout follow
                if price > high_20[i] and vol > vol_threshold:  # upward breakout
                    position = 1
                    signals[i] = position_size
                elif price < low_20[i] and vol > vol_threshold:  # downward breakdown
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
        elif position == 1:
            # Exit long: opposite band touch or regime shift against position
            if chop_val > 61.8 and price >= high_20[i]:  # in range, hit upper band -> exit
                position = 0
                signals[i] = 0.0
            elif chop_val < 38.2 and price < low_20[i]:  # in trend, break below lower band -> exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: opposite band touch or regime shift against position
            if chop_val > 61.8 and price <= low_20[i]:  # in range, hit lower band -> exit
                position = 0
                signals[i] = 0.0
            elif chop_val < 38.2 and price > high_20[i]:  # in trend, break above upper band -> exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Chop_Donchian_Volume_Regime"
timeframe = "4h"
leverage = 1.0