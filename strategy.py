#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for EMA trend and Camarilla levels.
- EMA34 > rising: bullish trend bias, EMA34 < falling: bearish trend bias.
- Entry: Long when price breaks above Camarilla H3 AND EMA34 trending up.
         Short when price breaks below Camarilla L3 AND EMA34 trending down.
- Exit: Opposite Camarilla breakout (L3 for long, H3 for short) or EMA trend reversal.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via breakouts with trend, works in bear via short breakouts with trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    # EMA trend: rising if current > previous, falling if current < previous
    ema_trend = np.zeros_like(ema_34)
    ema_trend[1:] = np.where(ema_34[1:] > ema_34[:-1], 1, np.where(ema_34[1:] < ema_34[:-1], -1, 0))
    # Align EMA trend to 12h
    ema_trend_aligned = align_htf_to_ltf(prices, df_1d, ema_trend)
    
    # Calculate Camarilla levels (H3, L3) from 1d OHLC
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 4
    camarilla_l3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 4
    camarilla_h3_vals = camarilla_h3.values
    camarilla_l3_vals = camarilla_l3.values
    # Align Camarilla levels to 12h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_vals)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_vals)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, 20)  # Need enough 1d bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_trend_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_trend_val = ema_trend_aligned[i]
        camarilla_h3_val = camarilla_h3_aligned[i]
        camarilla_l3_val = camarilla_l3_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if volume_spike[i]:
                if ema_trend_val > 0:  # Bullish trend: look for long breakout above H3
                    if curr_high > camarilla_h3_val:
                        signals[i] = 0.25
                        position = 1
                elif ema_trend_val < 0:  # Bearish trend: look for short breakdown below L3
                    if curr_low < camarilla_l3_val:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR EMA trend turns bearish
            if curr_low < camarilla_l3_val or ema_trend_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR EMA trend turns bullish
            if curr_high > camarilla_h3_val or ema_trend_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_CamarillaH3L3_1dEMA34Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0