#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume confirmation.
# Long when price breaks above 20-bar high, price > 1d EMA50, volume > 2.0x 20-bar avg.
# Short when price breaks below 20-bar low, price < 1d EMA50, volume > 2.0x 20-bar avg.
# Exit on opposite Donchian breakout or ATR-based stoploss.
# Donchian channels provide robust trend-following structure proven on SOLUSDT.
# 1d EMA50 filters counter-trend trades in bear markets (2022 crash, 2025 range).
# Volume confirmation reduces false breakouts. Discrete sizing 0.30 minimizes fee churn.
# Timeframe: 4h as per experiment guidelines.

name = "4h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # ATR(14) for dynamic stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA50 and ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_volume_confirm = volume_confirm[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 20-bar high, price > 1d EMA50, volume spike
            if (curr_close > curr_highest_20 and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short: price breaks below 20-bar low, price < 1d EMA50, volume spike
            elif (curr_close < curr_lowest_20 and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: opposite breakout or ATR stoploss
            if (curr_close < curr_lowest_20 or  # opposite Donchian breakout
                curr_close < entry_price - 2.5 * curr_atr):  # ATR stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit conditions: opposite breakout or ATR stoploss
            if (curr_close > curr_highest_20 or  # opposite Donchian breakout
                curr_close > entry_price + 2.5 * curr_atr):  # ATR stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals