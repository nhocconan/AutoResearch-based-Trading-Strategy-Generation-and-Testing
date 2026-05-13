#!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band squeeze breakout with 1w trend filter and volume confirmation. Uses 1d ATR for dynamic position sizing (0.20 in low vol, 0.30 in high vol). Designed for BTC/ETH robustness: Bollinger squeeze identifies low volatility primed for breakout, 1w EMA50 ensures alignment with weekly trend, volume confirmation filters false breakouts. ATR-based sizing adapts to market conditions, reducing size in high volatility (bear markets) and increasing in low volatility (consolidation). Targets 12-37 trades/year on 6h timeframe.

name = "6h_BBandSqueeze_Breakout_1wEMA50_VolumeConfirm_ATRSizing_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Bollinger Bands (20, 2.0) on 6h data
    close_s = pd.Series(close)
    sma_20 = close_s.rolling(window=20, min_periods=20).mean().values
    std_20 = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2.0 * std_20
    lower_band = sma_20 - 2.0 * std_20
    bb_width = (upper_band - lower_band) / sma_20  # normalized width
    
    # Bollinger Band squeeze: width < 20-period average width
    avg_bb_width = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < avg_bb_width
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Calculate 1d ATR for dynamic position sizing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Normalize ATR to get volatility regime (0.5-2.0 range)
    atr_ma = pd.Series(atr_14_aligned).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_ma > 0, atr_14_aligned / atr_ma, 1.0)
    atr_ratio = np.clip(atr_ratio, 0.5, 2.0)  # bound between 0.5 and 2.0
    
    # Dynamic position size: 0.20 in low vol (ratio < 0.8), 0.30 in high vol (ratio > 1.2)
    base_size = 0.25
    size_multiplier = np.where(atr_ratio < 0.8, 0.8, np.where(atr_ratio > 1.2, 1.2, 1.0))
    position_size = base_size * size_multiplier
    position_size = np.clip(position_size, 0.20, 0.30)  # enforce limits
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or 
            np.isnan(avg_bb_width[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(position_size[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # ENTRY: Bollinger Band breakout with volume confirmation and weekly trend filter
            # LONG: price breaks above upper band, volume > 1.5x avg, price > weekly EMA50
            if (close[i] > upper_band[i] and 
                volume[i] > 1.5 * avg_volume[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                squeeze[i]):  # only trade after squeeze
                signals[i] = position_size[i]
                position = 1
            # SHORT: price breaks below lower band, volume > 1.5x avg, price < weekly EMA50
            elif (close[i] < lower_band[i] and 
                  volume[i] > 1.5 * avg_volume[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  squeeze[i]):  # only trade after squeeze
                signals[i] = -position_size[i]
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to middle band (SMA20) OR weekly trend fails
            if (close[i] <= sma_20[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size[i]
        elif position == -1:
            # EXIT SHORT: price returns to middle band (SMA20) OR weekly trend fails
            if (close[i] >= sma_20[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size[i]
    
    return signals