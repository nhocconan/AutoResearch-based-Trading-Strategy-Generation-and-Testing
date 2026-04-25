#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_HTFTrend_EMA50
Hypothesis: Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above upper Donchian channel in 1d uptrend with volume > 1.8x 20-period average.
Short when price breaks below lower Donchian channel in 1d downtrend with volume > 1.8x 20-period average.
Uses discrete sizing (0.25) and ATR trailing stop (2.0) to limit trades (~20-50/year) and minimize fee drag.
Designed for BTC/ETH to work in bull/bear via breakout structure with trend and volume filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous day's Donchian channels (20-period)
    # Using 20-period lookback on 1d data
    lookback = 20
    upper_donchian = np.full(len(high_1d), np.nan)
    lower_donchian = np.full(len(low_1d), np.nan)
    
    for i in range(lookback, len(high_1d)):
        upper_donchian[i] = np.max(high_1d[i-lookback:i])
        lower_donchian[i] = np.min(low_1d[i-lookback:i])
    
    # Align Donchian levels to 4h timeframe
    upper_donchian_aligned = align_htf_to_ltf(prices, df_1d, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_1d, lower_donchian)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # ATR for stop loss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need EMA50 (50), Donchian (20+20=40), volume MA (20), ATR (14)
    start_idx = max(50, 40, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_donchian_aligned[i]) or np.isnan(lower_donchian_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian in 1d uptrend with volume spike
            long_signal = (curr_close > upper_donchian_aligned[i]) and \
                         (close_1d_aligned := align_htf_to_ltf(prices, df_1d, close_1d)[i]) > ema_50_1d_aligned[i] and \
                         volume_spike[i]
            # Short: price breaks below lower Donchian in 1d downtrend with volume spike
            short_signal = (curr_close < lower_donchian_aligned[i]) and \
                          (close_1d_aligned := align_htf_to_ltf(prices, df_1d, close_1d)[i]) < ema_50_1d_aligned[i] and \
                          volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below lower Donchian OR trend turns down OR ATR stoploss hit
            if (curr_close < lower_donchian_aligned[i]) or \
               (close_1d_aligned := align_htf_to_ltf(prices, df_1d, close_1d)[i]) < ema_50_1d_aligned[i] or \
               (curr_close < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian OR trend turns up OR ATR stoploss hit
            if (curr_close > upper_donchian_aligned[i]) or \
               (close_1d_aligned := align_htf_to_ltf(prices, df_1d, close_1d)[i]) > ema_50_1d_aligned[i] or \
               (curr_close > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_HTFTrend_EMA50"
timeframe = "4h"
leverage = 1.0