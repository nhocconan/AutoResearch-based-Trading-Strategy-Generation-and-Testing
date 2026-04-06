#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA + 1d RSI + volume confirmation
# Long when KAMA rising, RSI(1d) > 50, and volume > 1.5x 20-period average
# Short when KAMA falling, RSI(1d) < 50, and volume > 1.5x 20-period average
# Exit when KAMA changes direction (trend reversal)
# Uses 12h timeframe to limit trades, 1d RSI for trend filter, volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_kama_1d_rsi_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) on 12h
    def kama(close, er_period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(close, np.nan)
        kama[er_period] = close[er_period]
        for i in range(er_period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close, er_period=10, fast=2, slow=30)
    
    # 1-day RSI(14) trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate RSI on daily close
    delta = np.diff(daily_close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi.values])
    
    # Align daily RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(kama_vals[i]) or np.isnan(rsi_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = kama_vals[i] > kama_vals[i-1]
        kama_falling = kama_vals[i] < kama_vals[i-1]
        
        # Check exits: KAMA changes direction
        if position == 1:  # long position
            if not kama_rising:  # KAMA falling or flat
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if not kama_falling:  # KAMA rising or flat
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with RSI filter and volume confirmation
            # Long: KAMA rising AND RSI > 50 AND volume confirmation
            if kama_rising and rsi_aligned[i] > 50 and volume[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling AND RSI < 50 AND volume confirmation
            elif kama_falling and rsi_aligned[i] < 50 and volume[i] > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
    
    return signals