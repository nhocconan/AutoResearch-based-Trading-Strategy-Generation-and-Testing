# 4H_KAMA_Trend_RSI_With_Volume_Filter
# Hypothesis: KAMA adapts to market noise, reducing whipsaw in ranging markets while capturing trends.
# Combined with RSI(14) for momentum confirmation and volume filter to avoid low-quality signals.
# Works in bull (trend-following) and bear (avoids false signals via KAMA adaptation).
# Target: 20-40 trades/year on 4h timeframe to minimize fee drag.

#!/usr/bin/env python3
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
    
    # Load daily data for KAMA and RSI - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (ER=10) on daily close
    close_daily = df_daily['close'].values
    change = np.abs(np.diff(close_daily, prepend=close_daily[0]))
    volatility = np.sum(np.abs(np.diff(close_daily)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_daily)
    kama[0] = close_daily[0]
    for i in range(1, len(close_daily)):
        kama[i] = kama[i-1] + sc[i] * (close_daily[i] - kama[i-1])
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align KAMA and RSI to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi)
    
    # Volume average (20-period) on 4h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA, RSI > 50, and volume confirmation
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA, RSI < 50, and volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back through KAMA
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

name = "4H_KAMA_Trend_RSI_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0