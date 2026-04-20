#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume and volatility filter
# - Calculate daily Donchian channels (20-period high/low)
# - Long when price breaks above upper band with volume > 1.8x 20-period average
# - Short when price breaks below lower band with volume > 1.8x 20-period average
# - Exit when price crosses back through 10-period SMA or volatility drops
# - Uses 1d for channel calculation and 1w for trend filter
# - Target: 20-30 trades per year per symbol (80-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-period SMA for exit
    sma_10 = pd.Series(close_1d).rolling(window=10, min_periods=10).mean().values
    
    # Calculate weekly trend filter (EMA 50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR for volatility filter (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 20-period average volume
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align Donchian levels and SMA to 1d timeframe (already aligned via get_htf_data)
    # Actually, since we're using 1d as primary timeframe, we can use values directly
    # But we need to align weekly EMA to daily
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(sma_10[i]) or \
           np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above upper Donchian + volume surge + price above weekly EMA
            if price > high_20[i] and vol > 1.8 * vol_ma[i] and price > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below lower Donchian + volume surge + price below weekly EMA
            elif price < low_20[i] and vol > 1.8 * vol_ma[i] and price < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below 10-period SMA OR volatility drops below 0.5x ATR
            if price < sma_10[i] or atr_14_aligned[i] < 0.5 * np.mean(atr_14_aligned[max(0, i-20):i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 10-period SMA OR volatility drops below 0.5x ATR
            if price > sma_10[i] or atr_14_aligned[i] < 0.5 * np.mean(atr_14_aligned[max(0, i-20):i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Volume_VolatilityFilter"
timeframe = "1d"
leverage = 1.0