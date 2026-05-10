# 6H_200EMA_Cross_Signal
# Hypothesis: For 6h timeframe, use 200-period EMA on daily data as trend filter.
# Long when price crosses above 200EMA with volume confirmation and price above 50EMA.
# Short when price crosses below 200EMA with volume confirmation and price below 50EMA.
# Exit when price crosses back over 50EMA in opposite direction.
# Uses daily EMA200 as primary trend filter (works in bull/bear by following longer trend).
# Target: 20-40 trades/year per symbol.

name = "6H_200EMA_Cross_Signal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h indicators
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # 50 EMA for entry/exit timing
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily 200 EMA as trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily 200 EMA to 6h
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50[i]) or np.isnan(vol_ma[i]) or np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        price_above_ema50 = close[i] > ema50[i]
        price_below_ema50 = close[i] < ema50[i]
        
        price_above_ema200 = close[i] > ema200_1d_aligned[i]
        price_below_ema200 = close[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # Enter long: price crosses above 200EMA + above 50EMA + volume
            if price_above_ema200 and price_above_ema50 and volume_confirm and close[i-1] <= ema200_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below 200EMA + below 50EMA + volume
            elif price_below_ema200 and price_below_ema50 and volume_confirm and close[i-1] >= ema200_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below 50EMA
            if price_below_ema50 and close[i-1] >= ema50[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above 50EMA
            if price_above_ema50 and close[i-1] <= ema50[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals