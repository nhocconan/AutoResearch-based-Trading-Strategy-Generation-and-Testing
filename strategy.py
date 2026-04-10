#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d volume confirmation + chop regime filter
# - Long when price breaks above Camarilla H3 (1d) AND volume > 1.8x 20-period average AND chop > 61.8 (range)
# - Short when price breaks below Camarilla L3 (1d) AND volume > 1.8x 20-period average AND chop > 61.8 (range)
# - Exit when price crosses Camarilla Pivot point (1d) OR opposite breakout occurs
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Camarilla pivots work well in ranging markets (chop > 61.8) which is common in 2025+ bear/range regime
# - Volume confirmation reduces false breakouts
# - Chop filter ensures we only trade in ranging conditions where mean reversion at pivots works

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Pre-compute 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Pre-compute 4h Chopiness Index (14-period) for regime filter
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = true_range(high, low, prev_close)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chopiness Index = 100 * log10(sum(TR,14) / (max(high,14) - min(low,14))) / log10(14)
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range_hl = max_high - min_low
    # Avoid division by zero
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(sum_tr / range_hl) / np.log10(14)
    chop_filter = chop > 61.8  # Range regime
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate typical price for Camarilla
    typical_price = (high_1d + low_1d + close_1d) / 3
    # Camarilla levels based on previous day
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First bar: use same day
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close_1d) / 3
    range_1d = prev_high - prev_low
    
    # Camarilla levels
    H3 = pivot + (range_1d * 1.1 / 4)
    L3 = pivot - (range_1d * 1.1 / 4)
    H4 = pivot + (range_1d * 1.1 / 2)
    L4 = pivot - (range_1d * 1.1 / 2)
    
    # Align HTF indicators to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 (1d) AND volume spike AND chop > 61.8 (range)
            if (close[i] > H3_aligned[i] and 
                volume_spike[i] and 
                chop_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 (1d) AND volume spike AND chop > 61.8 (range)
            elif (close[i] < L3_aligned[i] and 
                  volume_spike[i] and 
                  chop_filter[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses pivot point (1d) OR opposite breakout occurs
            exit_long = (position == 1 and 
                        (close[i] < pivot_aligned[i] or close[i] < L3_aligned[i]))
            exit_short = (position == -1 and 
                         (close[i] > pivot_aligned[i] or close[i] > H3_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals