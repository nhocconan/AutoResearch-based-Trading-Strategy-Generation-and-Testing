#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend and Camarilla levels.
- Trend: Price > EMA34 = bullish bias (long H3 breakout, short only on strong breakdown).
         Price < EMA34 = bearish bias (short L3 breakdown, long only on strong breakout).
- Entry: Long when price breaks above H3 AND volume > 1.5 * 20-period volume MA.
         Short when price breaks below L3 AND volume > 1.5 * 20-period volume MA.
         In strong counter-trend: only allow entries with volume > 2.0 * 20-period volume MA.
- Exit: Opposite Camarilla level (L3 for long, H3 for short) or volume drops below 1.0 * MA.
- Volume confirmation avoids false breakouts in low-liquidity periods.
- Discrete signal size: 0.25 to balance profit potential and drawdown control.
- Target: 50-150 total trades over 4 years (12-38/year) for 4h timeframe.
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
    
    # Calculate EMA34 on 1d
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla levels (H3, L3) on 1d
    # Camarilla: based on previous day's range
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    range_ = prev_high - prev_low
    
    # H3 = prev_close + range * 1.1/4
    # L3 = prev_close - range * 1.1/4
    h3 = prev_close + (range_ * 1.1 / 4)
    l3 = prev_close - (range_ * 1.1 / 4)
    
    # Align Camarilla levels to 4h
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3.values)
    
    # Volume confirmation: 20-period volume MA on 4h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    volume_strong = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, 20)  # Need EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema_34_aligned[i]
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Determine trend bias from EMA34
                bullish_bias = curr_close > ema_val
                bearish_bias = curr_close < ema_val
                
                # Bullish breakout above H3
                if curr_high > h3_val:
                    # In bullish bias: normal volume confirmation
                    # In bearish bias: require strong volume to counter trend
                    if bullish_bias or volume_strong[i]:
                        signals[i] = 0.25
                        position = 1
                # Bearish breakdown below L3
                elif curr_low < l3_val:
                    # In bearish bias: normal volume confirmation
                    # In bullish bias: require strong volume to counter trend
                    if bearish_bias or volume_strong[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR volume drops significantly
            if curr_low < l3_val or volume[i] < volume_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR volume drops significantly
            if curr_high > h3_val or volume[i] < volume_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0