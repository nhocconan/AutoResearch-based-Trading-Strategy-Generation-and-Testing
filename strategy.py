#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 1d trend filter and volume confirmation.
# Uses 1d trend (EMA34) to filter direction, 4h Bollinger Bands (20,2) for entry,
# and volume spike (>1.5x EMA34 volume) for confirmation. Designed to work in both bull and bear markets.
# Target: 20-50 trades/year per symbol to avoid fee drag.
name = "4h_BollingerBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d trend filter: 34-period EMA on close (more stable than EMA20)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h Bollinger Bands: 20-period SMA +- 2*std
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # 4h volume average for spike detection
    vol_ema_4h = pd.Series(volume).ewm(span=34, adjust=False, min_periods=34).mean().values
    vol_spike = np.where(vol_ema_4h > 0, volume / vol_ema_4h, 1.0) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(vol_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long breakout: price > upper band with volume spike in uptrend
            long_condition = (close[i] > upper_band[i]) and vol_spike[i] and uptrend
            # Short breakdown: price < lower band with volume spike in downtrend
            short_condition = (close[i] < lower_band[i]) and vol_spike[i] and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below middle band or trend turns down
            if (close[i] < sma_20[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above middle band or trend turns up
            if (close[i] > sma_20[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals