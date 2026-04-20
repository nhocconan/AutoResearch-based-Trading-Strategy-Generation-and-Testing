#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d price action with 1-week trend filter and volume confirmation
# - Use 1d closing price relative to 1-week SMA(50) for trend direction
# - Enter long when price closes above 1d EMA(20) in uptrend with volume > 1.3x 20-day average
# - Enter short when price closes below 1d EMA(20) in downtrend with volume > 1.3x 20-day average
# - Exit when price crosses opposite EMA(20) or ATR-based stop (1.5x ATR)
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)
# - Designed to work in both bull (trend following) and bear (mean reversion at extremes)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1-week SMA(50) for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate 1d EMA(20) for entry/exit
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate ATR for stop loss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: 20-day average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema_20_1d_aligned[i]) or np.isnan(sma_50_1w_aligned[i]) or \
           np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]  # Use 1d close for signal generation
        vol = volume_1d[i]
        
        # Determine trend: price vs 1w SMA50
        uptrend = price > sma_50_1w_aligned[i]
        downtrend = price < sma_50_1w_aligned[i]
        
        if position == 0:
            # Long entry: price above EMA20 in uptrend with volume surge
            if price > ema_20_1d_aligned[i] and uptrend and vol > 1.3 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price below EMA20 in downtrend with volume surge
            elif price < ema_20_1d_aligned[i] and downtrend and vol > 1.3 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below EMA20 OR ATR stop hit (1.5*ATR)
            if price < ema_20_1d_aligned[i] or price < entry_price - 1.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA20 OR ATR stop hit (1.5*ATR)
            if price > ema_20_1d_aligned[i] or price > entry_price + 1.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA20_1wTrendFilter_Volume"
timeframe = "1d"
leverage = 1.0