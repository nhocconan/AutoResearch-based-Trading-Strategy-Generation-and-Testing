# 24997: 4h_1d_bollinger_squeeze_breakout_v1
# Hypothesis: 4-hour Bollinger Band squeeze (low volatility) followed by breakout with volume confirmation.
# Long when price breaks above upper BB after squeeze (BB width < 20th percentile) with volume > 2x average.
# Short when price breaks below lower BB after squeeze with volume > 2x average.
# Exit when price returns to middle BB (20-period SMA).
# Uses 1-day trend filter (price > 50 EMA for longs, < 50 EMA for shorts) to avoid counter-trend trades.
# Designed for low-frequency, high-quality trades (<30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_bollinger_squeeze_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2.0 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Bollinger Bands (20, 2)
    sma_20 = np.full(n, np.nan)
    std_20 = np.full(n, np.nan)
    for i in range(20, n):
        sma_20[i] = np.mean(close[i-20:i])
        std_20[i] = np.std(close[i-20:i])
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Percentile rank of BB width (20-period lookback)
    bb_width_rank = np.full(n, np.nan)
    for i in range(40, n):  # Need 20 for BB + 20 for rank
        window = bb_width[i-20:i]
        if not np.all(np.isnan(window)):
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                bb_width_rank[i] = (np.sum(valid <= bb_width[i]) / len(valid)) * 100
    
    # Volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(bb_width_rank[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        squeeze = bb_width_rank[i] < 20  # Bollinger squeeze: low volatility
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to middle BB (20 SMA)
            if price <= sma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to middle BB (20 SMA)
            if price >= sma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper BB after squeeze with volume expansion and above 1d EMA50
            if price > upper_bb[i] and squeeze and vol_ratio > 2.0 and price > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower BB after squeeze with volume expansion and below 1d EMA50
            elif price < lower_bb[i] and squeeze and vol_ratio > 2.0 and price < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals