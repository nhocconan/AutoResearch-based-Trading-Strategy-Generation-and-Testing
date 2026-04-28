#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI mean reversion and volume confirmation.
# Uses 1d primary timeframe targeting 7-25 trades/year (30-100 total over 4 years).
# KAMA adapts to market efficiency: tracks trend in low noise, avoids whipsaws in high noise.
# RSI(14) < 30 for long, > 70 for short with price > SMA50 (bull) or < SMA50 (bear) filter.
# Volume spike (>1.8x 20-bar average) confirms momentum.
# Position size 0.25 balances return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn.
# Works in bull/bear via trend filter + mean reversion logic.

name = "1d_KAMA_RSI_Volume_MeanRev_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d KAMA (adaptive trend)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * 0.2 + 0.06) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d SMA50 for trend filter
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d volume spike: >1.8x 20-bar average volume
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > 1.8 * volume_ma_20
    
    # Align HTF indicators to 1d timeframe (already 1d, but using helper for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for SMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(sma_50_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to SMA50
        price_above_sma = close[i] > sma_50_aligned[i]
        price_below_sma = close[i] < sma_50_aligned[i]
        
        # Mean reversion conditions with volume confirmation
        long_entry = (rsi_aligned[i] < 30) and price_above_sma and volume_spike_aligned[i]
        short_entry = (rsi_aligned[i] > 70) and price_below_sma and volume_spike_aligned[i]
        
        # Exit conditions: RSI reverts to midpoint (50) or opposite extreme
        long_exit = rsi_aligned[i] > 50
        short_exit = rsi_aligned[i] < 50
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals