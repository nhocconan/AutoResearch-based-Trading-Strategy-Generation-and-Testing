#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band mean reversion with 4h trend filter (EMA50) and volume confirmation (>1.5x 20-period average)
# In ranging markets, price tends to revert to the mean after touching Bollinger Bands (±2σ)
# 4h EMA50 ensures we only take mean reversal trades in the direction of the higher timeframe trend
# Volume confirmation filters weak breakouts; discrete sizing (0.20) minimizes fee churn
# Target: 80-150 total trades over 4 years (20-37/year) on 1h timeframe

name = "1h_BBMeanRev_4hEMA50_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Bollinger Bands (20, 2) on 1h timeframe
    close_s = pd.Series(close)
    sma_20 = close_s.rolling(window=20, min_periods=20).mean().values
    std_20 = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Calculate 20-period average volume for confirmation (on 1h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 20)  # 4h EMA50, BB, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_sma = sma_20[i]
        curr_upper = upper_band[i]
        curr_lower = lower_band[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price returns to middle (SMA20) OR hits upper band
            if curr_close >= curr_sma or curr_close >= curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price returns to middle (SMA20) OR hits lower band
            if curr_close <= curr_sma or curr_close <= curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price touches/lower band + price above 4h EMA50 (uptrend) + volume confirmation
            if (curr_close <= curr_lower and 
                curr_close > curr_ema_4h and 
                vol_confirm):
                signals[i] = 0.20
                position = 1
            # Short entry: price touches/upper band + price below 4h EMA50 (downtrend) + volume confirmation
            elif (curr_close >= curr_upper and 
                  curr_close < curr_ema_4h and 
                  vol_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals