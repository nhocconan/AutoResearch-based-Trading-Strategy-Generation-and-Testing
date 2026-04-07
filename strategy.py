#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h KAMA with 1D Trend Filter and Volume Confirmation
# Hypothesis: KAMA adapts to market volatility, reducing whipsaw in ranging markets.
# We trade in direction of 1-day EMA(50) when KAMA confirms trend (price > KAMA for long,
# price < KAMA for short), with volume confirmation. This captures trends while avoiding
# false signals in low volatility. Target: 12-37 trades/year on 12h timeframe.
name = "12h_kama_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Kaufman Adaptive Moving Average (KAMA) on 12h timeframe
    # ER = Efficiency Ratio = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    # Smooth constant = [ER * (fastest - slowest) + slowest]^2
    # fastest = 2/(2+1) = 0.6667, slowest = 2/(30+1) = 0.0645
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of absolute changes
    # Pad arrays to match length
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(1, np.nan), volatility])
    
    # Calculate ER and smooth constant with proper handling
    er = np.zeros(n)
    sc = np.zeros(n)
    for i in range(10, n):
        if not np.isnan(change_padded[i]) and volatility_padded[i] > 0:
            er[i] = change_padded[i] / volatility_padded[i]
        else:
            er[i] = 0
        sc[i] = (er[i] * (0.6667 - 0.0645) + 0.0645) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 1-day EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_12h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(daily_ema_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA or trend turns bearish
            if close[i] < kama[i] or close[i] < daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA or trend turns bullish
            if close[i] > kama[i] or close[i] > daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation and KAMA alignment
            if vol_filter[i]:
                # Long: price above KAMA + price above 1D EMA
                if close[i] > kama[i] and close[i] > daily_ema_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price below KAMA + price below 1D EMA
                elif close[i] < kama[i] and close[i] < daily_ema_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals