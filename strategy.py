#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Donchian breakout with volume confirmation and 1w KAMA trend filter.
# Weekly Donchian channels (40-period high/low) provide strong breakout signals.
# Volume > 1.5x 20-day average filters weak moves.
# Weekly KAMA (ER=10) defines trend direction to avoid counter-trend trades.
# Designed for 1d timeframe to capture major trend moves with low trade frequency (<10/year).
# Entry: Price breaks above weekly Donchian high + volume spike + bullish weekly KAMA.
# Exit: Price breaks below weekly Donchian low OR bearish weekly KAMA.
# Uses strict conditions to limit trades and avoid overtrading in choppy markets.

name = "1d_WeeklyDonchian_KAMA_Volume"
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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly Donchian channels (40-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=40, min_periods=40).max().values
    donchian_low = pd.Series(low_1w).rolling(window=40, min_periods=40).min().values
    
    # Weekly KAMA (ER=10)
    close_1w = df_1w['close'].values
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w, prepend=close_1w[0])), axis=0) if len(close_1w) > 1 else np.zeros_like(close_1w)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(10+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align weekly indicators to daily
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Volume filter: volume > 1.5 * 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(kama_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high + volume spike + bullish weekly KAMA
            if (close[i] > donchian_high_aligned[i] and 
                volume_spike[i] and 
                close[i] > kama_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low + volume spike + bearish weekly KAMA
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < kama_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly Donchian low OR bearish weekly KAMA
            if (close[i] < donchian_low_aligned[i]) or (close[i] < kama_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly Donchian high OR bullish weekly KAMA
            if (close[i] > donchian_high_aligned[i]) or (close[i] > kama_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals