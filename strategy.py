#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Alligator (Jaw=TEETH=LIPS SMMA) identifies trendless markets when lines intertwine
# Trend: price > 1w EMA50 for long, price < 1w EMA50 for short
# Entry: Alligator lines diverge (JAW < TEETH < LIPS for long, JAW > TEETH > LIPS for short) + volume > 1.5x average
# Exit: Alligator lines re-intertwine or trend reversal
# Works in both bull/bear markets by capturing trends after consolidation
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "12h_WilliamsAlligator_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator (SMMA = Smoothed Moving Average)
    # Jaw: SMMA(median, 13, 8) - slowest
    # Teeth: SMMA(median, 8, 5) - medium
    # Lips: SMMA(median, 5, 3) - fastest
    median = (high + low) / 2.0
    median_s = pd.Series(median)
    
    def smma(values, period, shift):
        """Smoothed Moving Average"""
        sma = values.rolling(window=period, min_periods=period).mean()
        # SMMA: first value = SMA, then recursive smoothing
        smma_vals = np.full_like(values, np.nan, dtype=float)
        smma_vals[period-1] = sma.iloc[period-1]
        for i in range(period, len(values)):
            if not np.isnan(sma.iloc[i]):
                smma_vals[i] = (smma_vals[i-1] * (period-1) + sma.iloc[i]) / period
            else:
                smma_vals[i] = smma_vals[i-1]
        # Apply shift
        smma_vals = np.roll(smma_vals, shift)
        smma_vals[:shift] = np.nan
        return smma_vals
    
    jaw = smma(median_s, 13, 8)
    teeth = smma(median_s, 8, 5)
    lips = smma(median_s, 5, 3)
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 80  # Need enough data for Alligator SMMA
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator lines diverge upward (JAW < TEETH < LIPS) + bullish trend + volume spike
            if (jaw[i] < teeth[i] and teeth[i] < lips[i] and  # Lines diverging up
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator lines diverge downward (JAW > TEETH > LIPS) + bearish trend + volume spike
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and  # Lines diverging down
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator lines re-intertwine OR trend turns bearish
            if not (jaw[i] < teeth[i] and teeth[i] < lips[i]) or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator lines re-intertwine OR trend turns bullish
            if not (jaw[i] > teeth[i] and teeth[i] > lips[i]) or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals