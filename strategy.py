#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with 1w EMA50 trend filter and volume confirmation
# Bollinger Bands identify volatility expansion/contraction; breakouts signal strong momentum
# 1w EMA50 ensures alignment with higher timeframe trend (avoid counter-trend trades)
# Volume > 1.5x 20-period average confirms breakout strength
# Designed to capture sustained moves in both bull and bear markets with low trade frequency
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe

name = "1d_BollingerBreakout_1wEMA50_Volume"
timeframe = "1d"
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
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Bollinger Bands (20, 2.0) on 1d timeframe
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2.0 * std_20)
    lower_band = sma_20 - (2.0 * std_20)
    
    # Calculate 20-period average volume for confirmation (on 1d timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 1w EMA50 and Bollinger Bands warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_upper = upper_band[i]
        curr_lower = lower_band[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below middle band (SMA20) OR bearish close below SMA20
            if curr_close < sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above middle band (SMA20) OR bullish close above SMA20
            if curr_close > sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above upper band + volume confirmation + price above 1w EMA50 (uptrend)
            if (curr_close > curr_upper and 
                vol_confirm and 
                curr_close > curr_ema_1w):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band + volume confirmation + price below 1w EMA50 (downtrend)
            elif (curr_close < curr_lower and 
                  vol_confirm and 
                  curr_close < curr_ema_1w):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals