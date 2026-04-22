#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI and volume confirmation for daily momentum
# Uses KAMA to capture adaptive trend, RSI for momentum strength, and volume for confirmation
# Designed to work in both bull and bear markets by following daily trend direction
# Target: 15-25 trades/year per symbol (60-100 total) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (adaptive moving average) on daily close
    # KAMA parameters: fast=2, slow=30, lookback=10
    close_1d_series = pd.Series(close_1d)
    change = abs(close_1d_series.diff(10))  # 10-period net change
    volatility = abs(close_1d_series.diff(1)).rolling(window=10, min_periods=10).sum()  # 10-period volatility
    er = change / volatility.replace(0, np.nan)  # efficiency ratio
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros(len(close_1d))
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI on daily close (14-period)
    delta = close_1d_series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume spike filter (20-period on daily)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to daily timeframe (1d)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA + RSI > 50 + volume spike
            if (close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA + RSI < 50 + volume spike
            elif (close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses KAMA in opposite direction
            if position == 1:
                if close[i] < kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Volume_Session"
timeframe = "1d"
leverage = 1.0