#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour KAMA trend with 1-day RSI(14) filter and volume confirmation
# Long when KAMA is rising, RSI(14) < 40 (oversold in uptrend), and volume > 1.5x 12h average volume
# Short when KAMA is falling, RSI(14) > 60 (overbought in downtrend), and volume > 1.5x 12h average volume
# Exit when KAMA direction changes or opposite signal occurs
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1d RSI for mean reversion filter within 12h trend
# Target: 50-150 total trades over 4 years (12-37/year)
# Designed to work in both bull and bear markets by combining trend following with mean reversion entries

name = "12h_kama_1d_rsi_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for KAMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate Efficiency Ratio for KAMA
    change = np.abs(np.diff(close_12h, k=10))  # 10-period change
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.abs(np.diff(close_12h, k=1))
    volatility = np.concatenate([np.array([np.nan]), volatility])
    vol_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = np.where(vol_sum > 0, change / vol_sum, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_12h, np.nan)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_12h = kama
    kama_12h_rising = kama_12h > np.roll(kama_12h, 1)
    kama_12h_falling = kama_12h < np.roll(kama_12h, 1)
    kama_12h_rising[0] = False
    kama_12h_falling[0] = False
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    kama_rising_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_rising.astype(float))
    kama_falling_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_falling.astype(float))
    
    # 1d data for RSI(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 12h volume average for confirmation
    volume_12h = df_12h['volume'].values
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(kama_rising_aligned[i]) or 
            np.isnan(kama_falling_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_ma_12h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA turns down or RSI overbought
            elif kama_falling_aligned[i] > 0.5 or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA turns up or RSI oversold
            elif kama_rising_aligned[i] > 0.5 or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: KAMA rising, RSI oversold (<40), volume spike
            if (kama_rising_aligned[i] > 0.5 and
                rsi_1d_aligned[i] < 40 and
                volume[i] > 1.5 * volume_ma_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: KAMA falling, RSI overbought (>60), volume spike
            elif (kama_falling_aligned[i] > 0.5 and
                  rsi_1d_aligned[i] > 60 and
                  volume[i] > 1.5 * volume_ma_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals