# 4h_1d_KAMA_Trend_Signal_v1
# Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) on 1d to detect trend direction.
# Long when 4h price closes above 1d KAMA, short when below, with volume confirmation.
# KAMA adapts to market noise - slows in ranging markets, speeds in trending markets.
# This should reduce whipsaws in ranging markets while capturing trends.
# Volume confirmation ensures breakouts have institutional participation.
# Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag.
# Works in bull via trend following, in bear via avoiding false signals during ranging periods.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_Trend_Signal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Parameters: ER length = 10, Fast SC = 2/(2+1), Slow SC = 2/(30+1)
    close_1d = df_1d['close'].values
    direction = np.abs(np.diff(close_1d, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)  # 10-period sum of abs changes
    volatility = np.append(np.full(9, np.nan), volatility)  # align with direction
    
    # Efficiency Ratio
    er = np.where(volatility > 0, direction / volatility, 0)
    # Smoothing Constants
    fast_sc = 2 / (2 + 1)      # EMA(2)
    slow_sc = 2 / (30 + 1)     # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: price vs KAMA with volume filter
        long_signal = close[i] > kama_aligned[i] and vol_ratio[i] > 1.3
        short_signal = close[i] < kama_aligned[i] and vol_ratio[i] > 1.3
        
        # Exit conditions: reverse signal
        exit_long = close[i] < kama_aligned[i]
        exit_short = close[i] > kama_aligned[i]
        
        # Signal logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals