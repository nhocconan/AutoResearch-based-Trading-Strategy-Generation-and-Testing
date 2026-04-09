#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA trend filter + volume confirmation
# Donchian breakout provides clear entry/exit levels with built-in structure
# 12h HMA(21) filters for higher timeframe trend alignment to avoid counter-trend trades
# Volume confirmation (current > 1.5 * 20-period average) ensures institutional participation
# Position size 0.25 to limit drawdown during 2022 bear market
# Target: 100-180 total trades over 4 years (25-45/year) to balance opportunity and fee drag
# Works in bull markets via breakouts, works in bear via short breakdowns + trend filter

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h HMA(21) for trend filter
    close_12h = df_12h['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    # WMA function for HMA calculation
    def wma(data, window):
        weights = np.arange(1, window + 1)
        return np.convolve(data, weights, mode='valid') / weights.sum()
    
    # Calculate HMA: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = np.full(len(close_12h), np.nan)
    wma_full = np.full(len(close_12h), np.nan)
    
    for i in range(half_len, len(close_12h)):
        wma_half[i] = wma(close_12h[i-half_len+1:i+1], half_len)
    for i in range(21, len(close_12h)):
        wma_full[i] = wma(close_12h[i-21+1:i+1], 21)
    
    hma_12h = np.full(len(close_12h), np.nan)
    for i in range(21 + sqrt_len - 1, len(close_12h)):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            raw_hma = 2 * wma_half[i] - wma_full[i]
            hma_12h[i] = wma(raw_hma, sqrt_len)
    
    # Align 12h HMA to 4h timeframe
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20+1:i+1])
        donchian_low[i] = np.min(low[i-20+1:i+1])
    
    # Calculate 4h volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(hma_12h_aligned[i]) or
            np.isnan(vol_ma[i]) or
            vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions: price breaks below Donchian low OR trend turns bearish
            if close[i] <= donchian_low[i] or close[i] < hma_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Donchian high OR trend turns bullish
            if close[i] >= donchian_high[i] or close[i] > hma_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout + volume confirmation + trend filter
            if vol_confirmed:
                # Long: price breaks above Donchian high in bullish trend
                if close[i] > donchian_high[i] and close[i] > hma_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low in bearish trend
                elif close[i] < donchian_low[i] and close[i] < hma_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals