#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout with 1d EMA34 Trend Filter and Volume Spike
Hypothesis: Camarilla H3/L3 levels act as strong intraday support/resistance. 
Breakouts above H3 or below L3 with 1d EMA34 trend alignment and volume confirmation 
capture momentum moves in both bull and bear markets. Using 12h timeframe reduces 
trade frequency to target 12-37 trades/year, minimizing fee drag while maintaining 
enough samples for statistical significance. The 1d EMA34 provides a stable trend 
filter that works across market regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Need to align 1d OHLC to 12h bars - get previous completed 1d bar's OHLC
    df_1d_ohLC = get_htf_data(prices, '1d')[['open', 'high', 'low', 'close']]
    if len(df_1d_ohLC) < 2:
        return np.zeros(n)
    
    # Shift to get previous day's OHLC (lookback 1 to avoid look-ahead)
    prev_high = df_1d_ohLC['high'].shift(1).values
    prev_low = df_1d_ohLC['low'].shift(1).values
    prev_close = df_1d_ohLC['close'].shift(1).values
    
    # Align previous day's OHLC to 12h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d_ohLC, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d_ohLC, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d_ohLC, prev_close)
    
    # Calculate Camarilla levels: H3, L3, H4, L4
    # H3 = Close + (High - Low) * 1.1 / 4
    # L3 = Close - (High - Low) * 1.1 / 4
    # H4 = Close + (High - Low) * 1.1 / 2
    # L4 = Close - (High - Low) * 1.1 / 2
    rang = prev_high_aligned - prev_low_aligned
    camarilla_h3 = prev_close_aligned + rang * 1.1 / 4
    camarilla_l3 = prev_close_aligned - rang * 1.1 / 4
    camarilla_h4 = prev_close_aligned + rang * 1.1 / 2
    camarilla_l4 = prev_close_aligned - rang * 1.1 / 2
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 and volume MA
    start_idx = max(34, 20)  # EMA34 lookback, volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Camarilla breakout + trend + volume
            # Long: price breaks above Camarilla H3 AND bullish bias AND volume spike
            long_entry = (curr_high > camarilla_h3[i]) and bullish_bias and vol_spike
            # Short: price breaks below Camarilla L3 AND bearish bias AND volume spike
            short_entry = (curr_low < camarilla_l3[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Camarilla L3 (mean reversion) OR loss of bullish bias
            if (curr_low < camarilla_l3[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Camarilla H3 (mean reversion) OR loss of bearish bias
            if (curr_high > camarilla_h3[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0