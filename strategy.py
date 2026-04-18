#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_Volume
Hypothesis: Trade Donchian(20) breakouts on daily timeframe with weekly trend filter and volume confirmation.
Long when price breaks above 20-day high AND weekly EMA34 trend is up (price > weekly EMA34) AND volume > 1.5x 20-day average volume.
Short when price breaks below 20-day low AND weekly EMA34 trend is down (price < weekly EMA34) AND volume > 1.5x 20-day average volume.
Exit when price crosses weekly EMA34 in opposite direction or volatility contraction (ATR < 0.5x 20-day ATR average).
Designed for low frequency (~10-25 trades/year) to minimize fee drag while capturing major trends in bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    ema_period = 34
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period-1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 / (ema_period + 1)) + (ema_1w[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align weekly EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    if len(high) >= lookback:
        for i in range(lookback-1, len(high)):
            highest_high[i] = np.max(high[i-lookback+1:i+1])
            lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: volume > 1.5x 20-day average volume
    vol_lookback = 20
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= vol_lookback:
        for i in range(vol_lookback, len(volume)):
            vol_ma[i] = np.mean(volume[i-vol_lookback:i])
    
    # ATR for volatility-based exit
    atr_period = 14
    tr = np.zeros_like(close)
    atr = np.full_like(close, np.nan)
    
    if len(high) >= 2:
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if len(tr) >= atr_period:
            atr[atr_period-1] = np.mean(tr[:atr_period])
            for i in range(atr_period, len(tr)):
                atr[i] = (tr[i] * 2 / (atr_period + 1)) + (atr[i-1] * (atr_period - 1) / (atr_period + 1))
    
    # ATR average for volatility filter
    atr_lookback = 20
    atr_ma = np.full_like(atr, np.nan)
    if len(atr) >= atr_lookback:
        for i in range(atr_lookback, len(atr)):
            atr_ma[i] = np.mean(atr[i-atr_lookback:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, vol_lookback, atr_lookback, ema_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_1w_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility filter: avoid trading in extremely low volatility
        vol_filter = atr[i] > 0.5 * atr_ma[i]
        
        if position == 0:
            # Long: price breaks above 20-day high AND weekly trend up AND volume confirmation AND volatility filter
            if close[i] > highest_high[i-1] and close[i] > ema_1w_aligned[i] and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND weekly trend down AND volume confirmation AND volatility filter
            elif close[i] < lowest_low[i-1] and close[i] < ema_1w_aligned[i] and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly EMA OR volatility contraction
            if close[i] < ema_1w_aligned[i] or not vol_filter:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly EMA OR volatility contraction
            if close[i] > ema_1w_aligned[i] or not vol_filter:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0