#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike_ATRStop
Hypothesis: Daily timeframe reduces trade frequency to avoid fee drag while capturing major trends. Uses Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation (>2.0x 20-bar avg volume). Long when price breaks above upper Donchian in 1w uptrend with volume spike; short when breaks below lower Donchian in 1w downtrend with volume spike. ATR-based stoploss (2.5x ATR). Designed for BTC/ETH to work in bull/bear via structure with trend/volume filters. Target trades: 30-100 total over 4 years.
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
    
    # 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_donchian = high_roll
    lower_donchian = low_roll
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr0 = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need EMA50 (50), Donchian (20), volume MA (20), ATR (14)
    start_idx = max(50, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian in 1w uptrend with volume spike
            long_signal = (curr_close > upper_donchian[i]) and \
                         (ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]) and \
                         volume_spike[i]
            # Short: price breaks below lower Donchian in 1w downtrend with volume spike
            short_signal = (curr_close < lower_donchian[i]) and \
                          (ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]) and \
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
            if (curr_close < lower_donchian[i]) or \
               (ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]) or \
               (curr_close < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian OR trend turns up OR ATR stoploss hit
            if (curr_close > upper_donchian[i]) or \
               (ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]) or \
               (curr_close > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0