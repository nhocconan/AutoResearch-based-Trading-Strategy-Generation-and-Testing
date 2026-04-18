#!/usr/bin/env python3
"""
1d_1w_Chop_Channel_Breakout
Hypothesis: Use 1w Donchian channel breakout with chop filter and volume confirmation on 1d timeframe.
In choppy markets (CHOP > 61.8), fade extreme moves (sell near upper band, buy near lower band).
In trending markets (CHOP < 38.2), breakout in direction of trend (buy above upper band, sell below lower band).
The chop filter adapts to market regime, reducing whipsaws in ranging markets and capturing trends.
Target: 15-25 trades/year by requiring weekly channel breakout, volume > 1.5x average, and chop regime alignment.
Works in bull markets by buying breakouts above weekly high, in bear markets by selling breakdowns below weekly low.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and chop calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    donch_high = np.full_like(close_1w, np.nan)
    donch_low = np.full_like(close_1w, np.nan)
    
    for i in range(20, len(close_1w)):
        donch_high[i] = np.max(high_1w[i-20:i])
        donch_low[i] = np.min(low_1w[i-20:i])
    
    # Calculate weekly chopiness index (14-period)
    atr_1w = np.full_like(close_1w, np.nan)
    for i in range(1, len(close_1w)):
        tr = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
        if i == 1:
            atr_1w[i] = tr
        else:
            atr_1w[i] = 0.9 * atr_1w[i-1] + 0.1 * tr  # Wilder's smoothing
    
    chop = np.full_like(close_1w, np.nan)
    for i in range(14, len(close_1w)):
        atr_sum = np.sum(atr_1w[i-14:i])
        donch_range = donch_high[i] - donch_low[i]
        if donch_range > 0:
            chop[i] = 100 * np.log10(atr_sum / donch_range) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral when no range
    
    # Align weekly indicators to daily timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need 20 for Donchian + 14 for chop
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine market regime based on chop value
            # CHOP > 61.8 = ranging/choppy (mean revert)
            # CHOP < 38.2 = trending (trend follow)
            if chop_aligned[i] > 61.8:
                # Choppy market: fade extreme moves
                if close[i] >= donch_high_aligned[i] and vol_confirm[i]:
                    signals[i] = -0.25  # sell at upper band
                    position = -1
                elif close[i] <= donch_low_aligned[i] and vol_confirm[i]:
                    signals[i] = 0.25   # buy at lower band
                    position = 1
                else:
                    signals[i] = 0.0
            else:
                # Trending market: breakout in direction of trend
                if close[i] > donch_high_aligned[i] and vol_confirm[i]:
                    signals[i] = 0.25   # buy breakout
                    position = 1
                elif close[i] < donch_low_aligned[i] and vol_confirm[i]:
                    signals[i] = -0.25  # sell breakdown
                    position = -1
                else:
                    signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to opposite Donchian band or chop increases significantly
            if (close[i] <= donch_low_aligned[i] or 
                chop_aligned[i] > 70):  # exit if market becomes too choppy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to opposite Donchian band or chop increases significantly
            if (close[i] >= donch_high_aligned[i] or 
                chop_aligned[i] > 70):  # exit if market becomes too choppy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Chop_Channel_Breakout"
timeframe = "1d"
leverage = 1.0