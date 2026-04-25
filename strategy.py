#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrend_RegimeFilter
Hypothesis: Camarilla R1/S1 breakouts on 4h with 12h EMA50 trend filter and choppiness regime filter. 
Only trade when 12h EMA50 confirms trend direction AND market is not too choppy (CHOP < 61.8). 
Volume confirmation ensures breakout validity. Discrete position sizing (0.25) minimizes fee churn. 
Designed for both bull/bear markets by following 12h trend and avoiding false breakouts in ranging markets.
Target: 20-40 trades/year to stay within fee drag limits.
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
    
    # Get 12h data for HTF trend filter and choppiness regime
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA50 on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels on 12h data (based on previous 12h bar's OHLC)
    camarilla_r1_12h = close_12h + ((high_12h - low_12h) * 1.1 / 12)
    camarilla_s1_12h = close_12h - ((high_12h - low_12h) * 1.1 / 12)
    camarilla_c_12h = close_12h  # Camarilla C is the close
    
    # Calculate Choppiness Index on 12h data (CHOP > 61.8 = ranging/choppy)
    # True Range = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    true_range[0] = tr1[0]  # First value
    
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = highest_high_14 - lowest_low_14
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    chop = 100 * np.log10(atr_14 * np.sqrt(14) / denominator) / np.log10(14)
    chop_regime = chop < 61.8  # True when NOT choppy (trending)
    
    # Align HTF indicators to 4h timeframe (completed 12h bar lag)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h, additional_delay_bars=1)
    camarilla_c_aligned = align_htf_to_ltf(prices, df_12h, camarilla_c_12h, additional_delay_bars=1)
    chop_regime_aligned = align_htf_to_ltf(prices, df_12h, chop_regime.astype(float), additional_delay_bars=1)
    
    # Volume confirmation: 1.5x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 and chop calculation
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_c_aligned[i]) or
            np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals in direction of 12h trend with volume confirmation and regime filter
            # Long: price breaks above R1 in uptrend (close > EMA50) AND not choppy
            # Short: price breaks below S1 in downtrend (close < EMA50) AND not choppy
            long_signal = (close[i] > camarilla_r1_aligned[i]) and \
                         (close[i] > ema50_aligned[i]) and \
                         volume_spike[i] and \
                         (chop_regime_aligned[i] > 0.5)
            short_signal = (close[i] < camarilla_s1_aligned[i]) and \
                          (close[i] < ema50_aligned[i]) and \
                          volume_spike[i] and \
                          (chop_regime_aligned[i] > 0.5)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Camarilla C (mean reversion to midpoint)
            exit_signal = close[i] < camarilla_c_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla C (mean reversion to midpoint)
            exit_signal = close[i] > camarilla_c_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hTrend_RegimeFilter"
timeframe = "4h"
leverage = 1.0