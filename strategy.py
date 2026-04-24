#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend direction.
- EMA34 > rising: bullish bias, EMA34 < falling: bearish bias.
- Entry: Long when price breaks above Camarilla H3 level AND EMA34 trending up AND volume > 1.5 * 20-period MA.
         Short when price breaks below Camarilla L3 level AND EMA34 trending down AND volume > 1.5 * 20-period MA.
- Exit: Opposite Camarilla level break (L3 for long, H3 for short) or EMA trend reversal.
- Volume confirmation: avoids false breakouts in low-volume environments.
- Discrete signal size: 0.25 to balance profit potential and drawdown control.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Get 1d data for Camarilla pivot calculation and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate Camarilla levels (H3, L3) from previous 1d bar
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # Using previous day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    camarilla_h3 = prev_close + 1.1 * prev_range / 2.0
    camarilla_l3 = prev_close - 1.1 * prev_range / 2.0
    
    # Align Camarilla levels to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate EMA34 on 1d for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # EMA trend direction: rising if current > previous, falling if current < previous
    ema_trend_up = np.zeros(len(ema_34_aligned), dtype=bool)
    ema_trend_down = np.zeros(len(ema_34_aligned), dtype=bool)
    ema_trend_up[1:] = ema_34_aligned[1:] > ema_34_aligned[:-1]
    ema_trend_down[1:] = ema_34_aligned[1:] < ema_34_aligned[:-1]
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(35, 20)  # Need enough 1d bars for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if volume_spike[i]:
                # Bullish breakout: price breaks above H3 AND EMA trending up
                if curr_high > camarilla_h3_aligned[i] and ema_trend_up[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below L3 AND EMA trending down
                elif curr_low < camarilla_l3_aligned[i] and ema_trend_down[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR EMA trend turns down
            if curr_low < camarilla_l3_aligned[i] or not ema_trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR EMA trend turns up
            if curr_high > camarilla_h3_aligned[i] or not ema_trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dEMA34Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0