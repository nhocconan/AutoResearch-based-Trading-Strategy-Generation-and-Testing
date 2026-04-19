#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d volume spike + 1d RSI momentum filter
# Donchian(20) breakout for trend direction and entry timing
# 1d volume spike (>1.5x 20-day average) for conviction
# 1d RSI(14) to avoid overextended moves (RSI < 70 for long, > 30 for short)
# Exit on opposite Donchian band touch or RSI mean reversion
# Designed to capture trend continuations with volume confirmation in both bull and bear markets
# Target: 20-35 trades/year to avoid fee drag
name = "4h_Donchian_1dVolume_RSI_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and RSI confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d RSI (14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 4h Donchian Channels (20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x average
        if i >= 20:
            vol_ma = vol_ma_1d_aligned[i]
        else:
            vol_ma = vol_ma_1d_aligned[i] if not np.isnan(vol_ma_1d_aligned[i]) else volume[i]
        volume_filter = vol_ma > 0 and volume[i] > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume + RSI not overbought
            if close[i] > donchian_high[i] and volume_filter and rsi_1d_aligned[i] < 70:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volume + RSI not oversold
            elif close[i] < donchian_low[i] and volume_filter and rsi_1d_aligned[i] > 30:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches Donchian low OR RSI < 30 (oversold)
            if close[i] < donchian_low[i] or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches Donchian high OR RSI > 70 (overbought)
            if close[i] > donchian_high[i] or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals