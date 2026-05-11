#!/usr/bin/env python3
# Weekly RSI(2) + Daily Close Below SMA200 + Volume Spike → Mean Reversion
# Hypothesis: Extreme weekly RSI(2) < 10 or > 90 signals exhaustion. Combined with daily price below SMA200 (bearish bias) for longs, or above for shorts, and volume spike for conviction, this captures mean-reversion bounces in both bull and bear markets. Weekly timeframe avoids noise; daily SMA200 filters for trend alignment. Volume surge confirms institutional interest. Designed for low trade frequency (<20/year) to minimize fee drag on 1d timeframe.
name = "1d_WeeklyRSI2_MeanReversion_SMA200_Volume"
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
    
    # Weekly RSI(2) for exhaustion signals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 3:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily SMA200 for trend filter
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Volume spike (2x 20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200  # Wait for SMA200
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(sma_200[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Weekly RSI(2) < 10 (oversold) + close < SMA200 (deep pullback in downtrend) + volume spike
            if rsi_1w_aligned[i] < 10 and close[i] < sma_200[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Weekly RSI(2) > 90 (overbought) + close > SMA200 (strong pullback in uptrend) + volume spike
            elif rsi_1w_aligned[i] > 90 and close[i] > sma_200[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Weekly RSI(2) > 50 (momentum shift) or close > SMA200 (trend resumption)
            if rsi_1w_aligned[i] > 50 or close[i] > sma_200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Weekly RSI(2) < 50 or close < SMA200
            if rsi_1w_aligned[i] < 50 or close[i] < sma_200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals