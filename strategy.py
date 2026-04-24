#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend and volume MA.
- Trend filter: EMA34 on 1d - price above EMA34 = bullish bias (long only), price below EMA34 = bearish bias (short only).
- Entry: Long when price breaks above Camarilla H3 level AND volume spike (volume > 1.5 * 20-period volume MA).
         Short when price breaks below Camarilla L3 level AND volume spike.
- Exit: Opposite Camarilla breakout (L3 for long, H3 for short) or volume drops below average.
- Uses discrete signal size 0.25 to limit drawdown and reduce fee churn.
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
    
    # Get 1d data for EMA trend and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 20-period volume MA on 1d
    volume_ma_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate Camarilla levels (H3, L3) from previous 1d bar
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # Using previous day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_h3_1d = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3_1d = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 34  # Need enough 1d bars for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema34 = ema34_aligned[i]
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        vol_ma = volume_ma_aligned[i]
        
        # Volume spike condition: current volume > 1.5 * 20-period volume MA
        volume_spike = curr_volume > (1.5 * vol_ma)
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike:
                if curr_close > ema34:  # Bullish bias: look for long
                    if curr_high > h3:  # Break above H3
                        signals[i] = 0.25
                        position = 1
                elif curr_close < ema34:  # Bearish bias: look for short
                    if curr_low < l3:  # Break below L3
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price drops below L3 OR volume drops below average
            if curr_low < l3 or curr_volume < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above H3 OR volume drops below average
            if curr_high > h3 or curr_volume < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0