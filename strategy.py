#!/usr/bin/env python3
"""
6h_IBS_Regime_Donchian_Breakout_v1
Hypothesis: Use 6h timeframe with IBS (Internal Bar Strength) to identify oversold/overbought conditions within the daily trend, combined with 1d EMA34 trend filter and volume confirmation. IBS = (close-low)/(high-low) captures intraday mean reversion tendency that works in both bull and bear markets when aligned with higher timeframe trend. Targets 12-30 trades/year to minimize fee drag.
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
    
    # Calculate IBS (Internal Bar Strength): (close-low)/(high-low)
    # Values near 0 = strong close (bearish), near 1 = strong close (bullish)
    ibs = (close - low) / (high - low + 1e-10)  # avoid division by zero
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for IBS calculation and indicators
    start_idx = max(20, 34, 20, 14)  # vol avg, 1d EMA, volume avg, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ibs[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for mean reversion entries aligned with 1d trend
            # Long: IBS < 0.3 (oversold) + 1d EMA34 uptrend + volume spike
            long_entry = (ibs[i] < 0.3) and \
                       (ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]) and \
                       volume_spike[i]
            # Short: IBS > 0.7 (overbought) + 1d EMA34 downtrend + volume spike
            short_entry = (ibs[i] > 0.7) and \
                        (ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when IBS > 0.7 (overbought) or ATR stoploss
            exit_condition = (ibs[i] > 0.7) or \
                           (close_val < entry_price - 2.0 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when IBS < 0.3 (oversold) or ATR stoploss
            exit_condition = (ibs[i] < 0.3) or \
                           (close_val > entry_price + 2.0 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_IBS_Regime_Donchian_Breakout_v1"
timeframe = "6h"
leverage = 1.0