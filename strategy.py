#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and 1d volume spike.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA34 trend direction, 1d for volume spike filter.
- Camarilla pivot levels: H3 = close + 1.1*(high-low)*1.1/12, L3 = close - 1.1*(high-low)*1.1/12.
- Trend Filter: Price > EMA34(4h) for long bias, Price < EMA34(4h) for short bias.
- Volume Confirmation: Current 1h volume > 2.0 * 24-period average 1d volume (scaled to 1h).
- Entry: Long when close crosses above H3 AND long bias AND volume confirmation.
         Short when close crosses below L3 AND short bias AND volume confirmation.
- Exit: Opposite Camarilla level (long exits when close < L3, short exits when close > H3).
- Signal size: 0.20 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with 4h trend and fading mean reversion extremes only with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate 1d volume average for confirmation (24-period, scaled to 1h equivalent)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 24:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_24_1d = pd.Series(df_1d['volume'].values).rolling(window=24, min_periods=24).mean().values
    vol_ma_24_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_24_1d)
    
    # Calculate Camarilla H3/L3 levels from daily data
    # H3 = close + 1.1*(high-low)*1.1/12, L3 = close - 1.1*(high-low)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla width multiplier: 1.1 * 1.1 / 12 = 1.21 / 12 = 0.100833
    camarilla_width = (high_1d - low_1d) * 0.100833
    h3_1d = close_1d + camarilla_width
    l3_1d = close_1d - camarilla_width
    
    # Align Camarilla levels to 1h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 24)  # Need 34 for EMA34, 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma_24_1d_aligned[i]) or
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price > EMA34(4h) for long bias, price < EMA34(4h) for short bias
        long_bias = curr_close > ema34_4h_aligned[i]
        short_bias = curr_close < ema34_4h_aligned[i]
        
        # Volume confirmation: current 1h volume > 2.0 * 24-period average 1d volume
        # Scale 1d volume to 1h equivalent: divide by 6 (since 6 * 1h = 1d)
        volume_confirm = curr_volume > 2.0 * (vol_ma_24_1d_aligned[i] / 6.0) if not np.isnan(vol_ma_24_1d_aligned[i]) else False
        
        # Camarilla breakout conditions
        crossed_above_h3 = (curr_close > h3_1d_aligned[i]) and (i == start_idx or close[i-1] <= h3_1d_aligned[i-1])
        crossed_below_l3 = (curr_close < l3_1d_aligned[i]) and (i == start_idx or close[i-1] >= l3_1d_aligned[i-1])
        
        # Exit conditions: opposite Camarilla level
        if position != 0:
            # Exit long: close crosses below L3
            if position == 1:
                if curr_close < l3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close crosses above H3
            elif position == -1:
                if curr_close > h3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume filters
        if position == 0:
            # Long: close crosses above H3 AND long bias AND volume confirmation
            long_condition = crossed_above_h3 and long_bias and volume_confirm
            
            # Short: close crosses below L3 AND short bias AND volume confirmation
            short_condition = crossed_below_l3 and short_bias and volume_confirm
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34Trend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0