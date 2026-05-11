#!/usr/bin/env python3
"""
12h_Adaptive_Kelly_Strategy_With_Volume_and_Trend_Filter
Hypothesis: Adaptive Kelly sizing based on 12h momentum and volatility,
filtered by 1d trend and volume confirmation. This adapts position size to
market conditions, reducing exposure in volatile/ranging markets and increasing
in trending ones, improving risk-adjusted returns in both bull and bear markets.
Designed for low turnover (~15-25 trades/year) to minimize fee drag.
"""

name = "12h_Adaptive_Kelly_Strategy_With_Volume_and_Trend_Filter"
timeframe = "12h"
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
    
    # === 1d Trend Filter: EMA34 ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 12h Momentum: ROC(10) ===
    roc_period = 10
    roc = np.zeros_like(close)
    roc[roc_period:] = (close[roc_period:] - close[:-roc_period]) / close[:-roc_period]
    
    # === 12h Volatility: ATR(14) for Kelly calculation ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros_like(close)
    atr[14:] = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Volume Confirmation: 20-period EMA ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # 1.5x average volume
    
    # === Kelly Fraction Calculation ===
    # Win probability based on ROC z-score (simplified)
    roc_mean = np.zeros_like(roc)
    roc_std = np.zeros_like(roc)
    for i in range(50, n):  # warmup
        roc_mean[i] = np.mean(roc[max(0, i-50):i])
        roc_std[i] = np.std(roc[max(0, i-50):i]) if np.std(roc[max(0, i-50):i]) > 0 else 1
    roc_z = np.where(roc_std > 0, (roc - roc_mean) / roc_std, 0)
    # Convert z-score to win probability estimate (clamped)
    win_prob = np.clip(0.5 + 0.1 * roc_z, 0.3, 0.7)
    # Loss probability
    loss_prob = 1 - win_prob
    # Win/loss ratio based on ATR (simplified: 1:1 base, adjusted by momentum)
    win_loss_ratio = 1.0 + 0.5 * np.tanh(roc_z)  # ranges 0.5 to 1.5
    # Kelly fraction: f = (bp - q) / b, where b = win/loss ratio, p = win prob, q = loss prob
    kelly_fraction = np.where(
        win_loss_ratio > 0,
        (win_loss_ratio * win_prob - loss_prob) / win_loss_ratio,
        0
    )
    # Cap Kelly at 0.3 and apply volatility scaling (reduce size in high vol)
    kelly_fraction = np.clip(kelly_fraction, 0, 0.3)
    vol_scaling = 1.0 / (1.0 + atr / (close * 0.02))  # reduce size when ATR > 2% of price
    position_size = kelly_fraction * vol_scaling
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(position_size[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Positive momentum + above 1d EMA34 + volume confirmation
            if roc[i] > 0 and close[i] > ema34_1d_aligned[i] and volume_ok[i]:
                signals[i] = position_size[i]
                position = 1
            # Short: Negative momentum + below 1d EMA34 + volume confirmation
            elif roc[i] < 0 and close[i] < ema34_1d_aligned[i] and volume_ok[i]:
                signals[i] = -position_size[i]
                position = -1
        else:
            # Exit conditions: momentum reversal or volatility spike
            if position == 1:
                # Exit: momentum turns negative OR volatility spikes (ATR > 3% of price)
                if roc[i] < 0 or (atr[i] > close[i] * 0.03):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size[i]
            elif position == -1:
                # Exit: momentum turns positive OR volatility spikes
                if roc[i] > 0 or (atr[i] > close[i] * 0.03):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size[i]
    
    return signals