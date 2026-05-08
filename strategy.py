# USING PROVEN PATTERNS FROM THE RESEARCH NOTES: CAMARILLA PIVOT + VOLUME SPIKE + CHOPPINESS REGIME
# This pattern achieved ETHUSDT test Sharpe 1.47 in the research notes.
# Adapting for 12h timeframe with 1d HTF as specified in the experiment.
# Camarilla levels provide precise support/resistance, volume confirms breakout strength,
# and chop filter avoids whipsaws in ranging markets.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Volume_Chop"
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
    
    # Calculate 12h Camarilla levels from previous day's OHLC
    # Using 1d data to get previous day's close, high, low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values  # Previous day's close
    prev_high = df_1d['high'].shift(1).values    # Previous day's high
    prev_low = df_1d['low'].shift(1).values      # Previous day's low
    
    # Calculate Camarilla levels for each 12h bar based on previous day's data
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # H2 = close + 0.6 * (high - low)
    # H1 = close + 0.275 * (high - low)
    # L1 = close - 0.275 * (high - low)
    # L2 = close - 0.6 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    
    range_1d = prev_high - prev_low
    H4 = prev_close + 1.5 * range_1d
    H3 = prev_close + 1.1 * range_1d
    H2 = prev_close + 0.6 * range_1d
    H1 = prev_close + 0.275 * range_1d
    L1 = prev_close - 0.275 * range_1d
    L2 = prev_close - 0.6 * range_1d
    L3 = prev_close - 1.1 * range_1d
    L4 = prev_close - 1.5 * range_1d
    
    # Align 1d Camarilla levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 12h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Choppiness Index (14-period) on 12h data
    # CHOP = 100 * log10(sum(TR over n) / (n * (max_high - min_low))) / log10(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First value
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr_sum / (14 * (max_high - min_low))) / np.log10(14)
    chop[np.isnan(chop)] = 50  # Default to middle range when undefined
    
    # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending (trend follow)
    chop_ranging = chop > 61.8
    chop_trending = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # In trending markets: breakout strategy
            # Long when price breaks above H4 with volume spike
            # Short when price breaks below L4 with volume spike
            long_cond = chop_trending[i] and (close[i] > H4_aligned[i]) and volume_filter[i]
            short_cond = chop_trending[i] and (close[i] < L4_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit conditions
            # Exit if: price drops below H3 (profit target) OR chop becomes ranging (avoid whipsaw)
            if chop_ranging[i] or close[i] < H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit conditions
            # Exit if: price rises above L3 (profit target) OR chop becomes ranging (avoid whipsaw)
            if chop_ranging[i] or close[i] > L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals