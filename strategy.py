#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ATR breakout with 1d trend filter and volume confirmation.
# In trending markets, price breaks beyond ATR-based channels with continuation.
# Uses 1d EMA34 for trend direction and volume spike for confirmation.
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 15-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on daily close
    ema_34_1d = np.full(len(df_1d), np.nan)
    alpha = 2 / (34 + 1)
    for i in range(len(close_1d)):
        if i < 33:
            ema_34_1d[i] = np.mean(close_1d[:i+1]) if i > 0 else close_1d[i]
        else:
            if np.isnan(ema_34_1d[i-1]):
                ema_34_1d[i] = np.mean(close_1d[i-33:i+1])
            else:
                ema_34_1d[i] = close_1d[i] * alpha + ema_34_1d[i-1] * (1 - alpha)
    
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) on 12h data
    atr = np.full(n, np.nan)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    for i in range(14, n):
        if np.isnan(atr[i-1]):
            atr[i] = np.nanmean(tr[i-13:i+1])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate upper and lower ATR bands (2 * ATR)
    upper_band = close + 2 * atr
    lower_band = close - 2 * atr
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(upper_band[i]) or
            np.isnan(lower_band[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from daily EMA34
        # Use previous bar's EMA to avoid look-ahead
        if i > 0 and not np.isnan(ema_34_1d_aligned[i-1]):
            trend_up = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            trend_down = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long entry: price breaks above upper ATR band + uptrend + volume spike
            if (close[i] > upper_band[i] and 
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower ATR band + downtrend + volume spike
            elif (close[i] < lower_band[i] and 
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price falls below lower ATR band or trend turns down
            if (close[i] < lower_band[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above upper ATR band or trend turns up
            if (close[i] > upper_band[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ATRBreakout_1dEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0