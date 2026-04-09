#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1w MACD regime filter
# Uses 6h Bull/Bear Power (EMA13) to measure buying/selling pressure
# 1w MACD histogram determines regime: bullish when MACD>signal, bearish when MACD<signal
# Long when Bull Power > 0 AND 1w MACD histogram rising (bullish regime)
# Short when Bear Power < 0 AND 1w MACD histogram falling (bearish regime)
# Volume confirmation: current volume > 1.5x 20-period average to avoid low-conviction moves
# Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear: MACD regime filter ensures alignment with higher timeframe momentum

name = "6h_1w_elder_ray_macd_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for MACD regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w MACD (12,26,9)
    close_1w = df_1w['close'].values
    ema12 = pd.Series(close_1w).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close_1w).ewm(span=26, adjust=False).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    macd_hist = macd_line - signal_line
    
    # Align 1w MACD histogram to 6h timeframe
    macd_hist_6h = align_htf_to_ltf(prices, df_1w, macd_hist)
    
    # Calculate 6h EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Buying pressure
    bear_power = low - ema13   # Selling pressure (negative values)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(macd_hist_6h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: Bull Power <= 0 OR MACD histogram turns negative (regime change)
            if bull_power[i] <= 0 or macd_hist_6h[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 OR MACD histogram turns positive (regime change)
            if bear_power[i] >= 0 or macd_hist_6h[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and MACD regime alignment
            if volume_confirm:
                # Long entry: Bull Power > 0 AND MACD histogram > 0 (bullish regime)
                if bull_power[i] > 0 and macd_hist_6h[i] > 0:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Bear Power < 0 AND MACD histogram < 0 (bearish regime)
                elif bear_power[i] < 0 and macd_hist_6h[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals