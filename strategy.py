#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume spike + chop regime filter
# - Camarilla pivot levels from 1d: L3/S3 for long, H3/H4 for short
# - Volume confirmation: 12h volume > 2.0x 20-period average (strong breakout filter)
# - Regime filter: 1d Choppiness Index > 61.8 for ranging market (mean reversion at pivots)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work well in ranging/mean-reverting markets (chop > 61.8)
# - Volume confirmation ensures breakouts have conviction
# - Chop regime filter avoids false signals in strong trends where pivots fail

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # L3 = C - (H - L) * 1.1 / 4
    # H3 = C + (H - L) * 1.1 / 4
    # L4 = C - (H - L) * 1.1 / 2
    # H4 = C + (H - L) * 1.1 / 2
    L3 = close_1d - (range_1d * 1.1 / 4)
    H3 = close_1d + (range_1d * 1.1 / 4)
    L4 = close_1d - (range_1d * 1.1 / 2)
    H4 = close_1d + (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    
    # Pre-compute 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log(n))) / log10(n)
    # Simplified: CHOP = 100 * log10(ATR_sum / (ATR_max * n)) / log10(n)
    # We'll use a common approximation: CHOP = 100 * log10(sum(TrueRange(14)) / (max(TrueRange(14)) * 14)) / log10(14)
    tr1 = pd.Series(high_1d).rolling(2).max().values - pd.Series(low_1d).rolling(2).min().values
    tr2 = abs(pd.Series(high_1d).shift(1).values - pd.Series(close_1d).values)
    tr3 = abs(pd.Series(low_1d).shift(1).values - pd.Series(close_1d).values)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Avoid division by zero
    true_range = np.where(true_range == 0, 1e-10, true_range)
    
    tr_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    tr_max = pd.Series(true_range).rolling(window=14, min_periods=14).max().values
    chop_raw = 100 * np.log10(tr_sum / (tr_max * 14)) / np.log10(14)
    chop_1d = chop_raw  # Already in 0-100 range
    
    # Align chop to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(L3_aligned[i]) or np.isnan(H3_aligned[i]) or
            np.isnan(L4_aligned[i]) or np.isnan(H4_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Regime filter: chop > 61.8 = ranging market (good for pivot mean reversion)
        chop_regime = chop_1d_aligned[i] > 61.8
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Entry conditions (only in choppy regime)
        enter_long = False
        enter_short = False
        
        if chop_regime and vol_confirm:
            # Long: price breaks above L3 (mean reversion long)
            if price_close > L3_aligned[i]:
                enter_long = True
            # Short: price breaks below H3 (mean reversion short)
            elif price_close < H3_aligned[i]:
                enter_short = True
        
        # Exit conditions: opposite pivot level or regime change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L4 (stop) or regime turns trending
            exit_long = (price_close < L4_aligned[i]) or (chop_1d_aligned[i] <= 61.8)
        elif position == -1:
            # Exit short if price breaks above H4 (stop) or regime turns trending
            exit_short = (price_close > H4_aligned[i]) or (chop_1d_aligned[i] <= 61.8)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals