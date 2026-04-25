#!/usr/bin/env python3
"""
12h_CCI_MeanReversion_1dTrendFilter_VolumeSpike_v1
Hypothesis: Trade 12h CCI extreme reversals aligned with daily EMA50 trend and volume spike (>1.5*ATR14).
Uses CCI(20) for mean reversion: long when CCI < -100, short when CCI > +100.
Daily trend filter ensures we trade with higher timeframe momentum, reducing whipsaws.
Volume confirmation adds conviction to reversals. Discrete sizing 0.25 limits fee drag.
Target: 12-37 trades/year to avoid fee drag while maintaining edge in both bull and bear markets.
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
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR14 for volume confirmation
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate CCI(20)
    lookback = 20
    tp = (high + low + close) / 3.0
    ma_tp = pd.Series(tp).rolling(window=lookback, min_periods=lookback).mean().values
    md_tp = pd.Series(tp).rolling(window=lookback, min_periods=lookback).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - ma_tp) / (0.015 * md_tp)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for daily EMA50, ATR, and CCI
    start_idx = max(50, 14, lookback)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(cci[i]) or np.isnan(ma_tp[i]) or np.isnan(md_tp[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * ATR
        volume_confirm = volume[i] > 1.5 * atr[i]
        
        # Determine daily trend from EMA50
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)[i]
        if np.isnan(daily_close_aligned):
            signals[i] = 0.0
            continue
            
        if daily_close_aligned > ema_50_1d_aligned[i]:
            daily_trend = 'bullish'  # allow longs
        elif daily_close_aligned < ema_50_1d_aligned[i]:
            daily_trend = 'bearish'  # allow shorts
        else:
            daily_trend = 'neutral'  # no trades in neutral zone
        
        if position == 0:
            # Long setup: CCI < -100 (oversold) AND volume confirm AND bullish daily trend
            long_setup = (cci[i] < -100) and volume_confirm and (daily_trend == 'bullish')
            
            # Short setup: CCI > +100 (overbought) AND volume confirm AND bearish daily trend
            short_setup = (cci[i] > 100) and volume_confirm and (daily_trend == 'bearish')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: CCI crosses above zero (mean reversion complete) OR daily trend turns bearish
            if (cci[i] > 0) or (daily_trend == 'bearish'):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: CCI crosses below zero (mean reversion complete) OR daily trend turns bullish
            if (cci[i] < 0) or (daily_trend == 'bullish'):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_CCI_MeanReversion_1dTrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0