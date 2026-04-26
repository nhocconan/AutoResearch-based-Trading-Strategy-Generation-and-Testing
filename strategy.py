#!/usr/bin/env python3
"""
6h_WilliamsVixFix_Breakout_1wTrend_v1
Hypothesis: Williams Vix Fix (WVF) identifies volatility spikes and potential reversals. 
Trade breakouts of Donchian(20) channels in the direction of weekly trend (EMA50) with WVF confirmation.
Works in bull markets by catching momentum bursts and in bear markets by fading extreme fear spikes.
Target: 12-30 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for HTF trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Williams Vix Fix: measures volatility, high = fear
    # WVF = ((Highest Close in period - Low) / Highest Close in period) * 100
    lookback = 22  # ~1 month for 6h charts
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    wvf = ((highest_close - low) / np.where(highest_close != 0, highest_close, 1)) * 100
    # WVF > 80 indicates extreme fear (potential buying opportunity)
    
    # Donchian(20) channels for breakouts
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of weekly EMA (50), WVF lookback (22), Donchian (20)
    start_idx = max(50, 22, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(wvf[i]) or
            np.isnan(donchian_h[i]) or
            np.isnan(donchian_l[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        wvf_val = wvf[i]
        
        if position == 0:
            # Long: Donchian breakout above + weekly uptrend + extreme fear (WVF > 80) for contrarian entry
            long_signal = (close_val > donchian_h[i]) and \
                          (close_val > ema_50_1w_val) and \
                          (wvf_val > 80)
            
            # Short: Donchian breakdown below + weekly downtrend + low fear (WVF < 20) for continuation
            short_signal = (close_val < donchian_l[i]) and \
                           (close_val < ema_50_1w_val) and \
                           (wvf_val < 20)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below Donchian low OR weekly trend turns bearish
            if (close_val < donchian_l[i]) or \
               (close_val < ema_50_1w_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian high OR weekly trend turns bullish
            if (close_val > donchian_h[i]) or \
               (close_val > ema_50_1w_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsVixFix_Breakout_1wTrend_v1"
timeframe = "6h"
leverage = 1.0