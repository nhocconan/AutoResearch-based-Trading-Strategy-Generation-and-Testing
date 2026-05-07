#!/usr/bin/env python3
"""
4h_Williams_VixFix_MeanReversion
Hypothesis: Williams VixFix identifies volatility spikes during panic selling in bear markets and complacency in bull markets. Combined with 1-week trend filter and Bollinger mean reversion, it captures oversold bounces in downtrends and overbought pullbacks in uptrends. Designed for 4h to achieve 20-35 trades/year with controlled risk, working in both bull and bear regimes by following higher timeframe trend while fading short-term extremes.
"""
name = "4h_Williams_VixFix_MeanReversion"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Williams VixFix: measures put/call panic via high-low relationship
    # Formula: VIXFIX = (HIGHEST HIGH - LOW) / HIGHEST HIGH * 100
    # We use 22-period lookback (approx 1 month) to match VIX calculation
    highest_high = pd.Series(high).rolling(window=22, min_periods=22).max().values
    vixfix = (highest_high - low) / highest_high * 100
    
    # Bollinger Bands for mean reversion signals
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # 1-week EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: avoid low-liquidity periods
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 0.5)  # at least half average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 70  # sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(vixfix[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VixFix spike (fear) + price below BB lower + 1w uptrend
            if (vixfix[i] > np.percentile(vixfix[max(0, i-100):i+1], 80) and  # recent high volatility
                close[i] < bb_lower[i] and
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: VixFix spike (fear) + price above BB upper + 1w downtrend
            elif (vixfix[i] > np.percentile(vixfix[max(0, i-100):i+1], 80) and  # recent high volatility
                  close[i] > bb_upper[i] and
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to Bollinger middle (mean reversion complete)
            if position == 1:
                if close[i] >= sma_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] <= sma_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals