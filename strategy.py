#!/usr/bin/env python3
"""
1d_Weekly_Momentum_v1
Hypothesis: Use weekly price momentum (4-week ROC) to identify strong trends, filtered by weekly trend (EMA34) and volume spikes.
In bull markets, strong momentum continues; in bear markets, extreme momentum often signals exhaustion and mean reversion.
Target: 15-30 trades per year on 1d timeframe (60-120 total over 4 years).
"""

name = "1d_Weekly_Momentum_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR MOMENTUM AND TREND ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly ROC(4) - 4-week rate of change (momentum)
    roc4_1w = np.full(len(close_1w), np.nan)
    for i in range(4, len(close_1w)):
        if close_1w[i-4] != 0:
            roc4_1w[i] = (close_1w[i] / close_1w[i-4] - 1) * 100  # percentage
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Weekly volume SMA20 for volume spike detection
    vol_sma20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all weekly data to daily
    roc4_1w_aligned = align_htf_to_ltf(prices, df_1w, roc4_1w)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    vol_sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(roc4_1w_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_sma20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike condition: current weekly volume > 1.5x 20-period average
        volume_spike = volume_1w[-1] > vol_sma20_1w_aligned[i] * 1.5 if len(volume_1w) > 0 else False
        # Simplified: use aligned volume data for current bar
        # Since we don't have direct weekly volume per daily bar, approximate with price-based volume filter
        vol_ratio = volume[i] / (np.mean(volume[max(0, i-20):i]) + 1e-10) if i >= 20 else 1.0
        volume_spike = vol_ratio > 2.0  # Daily volume > 2x 20-day average
        
        if position == 0:
            # Long: strong positive momentum (>5%) with volume spike AND price above weekly EMA34 (bullish continuation)
            # OR extremely negative momentum (<-15%) with volume spike AND price below weekly EMA34 (exhaustion bounce)
            if ((roc4_1w_aligned[i] > 5 and volume_spike and close[i] > ema34_1w_aligned[i]) or
                (roc4_1w_aligned[i] < -15 and volume_spike and close[i] < ema34_1w_aligned[i])):
                signals[i] = 0.25
                position = 1
            # Short: extremely negative momentum (<-15%) with volume spike AND price below weekly EMA34 (continuation)
            elif roc4_1w_aligned[i] < -15 and volume_spike and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: momentum turns negative (<-5%) or volume dries up
            if roc4_1w_aligned[i] < -5 or vol_ratio < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: momentum turns positive (>5%) or volume dries up
            if roc4_1w_aligned[i] > 5 or vol_ratio < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals