#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d Supertrend(10,3) filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Supertrend direction and Camarilla pivot levels (H3/L3).
- Camarilla Pivots: H3, L3 levels from prior 1d OHLC for breakout logic.
- Trend Filter: 1d Supertrend(10,3) must align with breakout direction.
- Volume Filter: Current 6h volume > 1.8 * 20-period average 6h volume.
- Entry: Long when close > H3 AND Supertrend=uptrend AND volume spike.
         Short when close < L3 AND Supertrend=downtrend AND volume spike.
- Exit: Opposite Camarilla break (long exits when close < L3, short exits when close > H3).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture strong momentum bursts aligned with daily trend while filtering chop/whipsaws.
- Works in bull markets (trend continuation) and bear markets (trend continuation down).
- Supertrend is more adaptive than EMA and reduces whipsaw in sideways markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivots (H3, L3) from prior day OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla H3 and L3 levels
    camarilla_range = prev_high - prev_low
    h3 = prev_close + camarilla_range * 1.1 / 2
    l3 = prev_close - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate 1d Supertrend(10,3)
    # True Range
    tr1 = pd.Series(df_1d['high']).shift(0) - pd.Series(df_1d['low']).shift(0)
    tr2 = abs(pd.Series(df_1d['high']).shift(0) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']).shift(0) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=10, adjust=False, min_periods=10).mean()
    
    # Basic Upper and Lower Bands
    hl2 = (pd.Series(df_1d['high']) + pd.Series(df_1d['low'])) / 2
    upper_basic = hl2 + (3 * atr)
    lower_basic = hl2 - (3 * atr)
    
    # Final Upper and Lower Bands
    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    
    for i in range(1, len(upper_basic)):
        if upper_basic[i] < upper_band[i-1] or df_1d['close'].iloc[i-1] > upper_band[i-1]:
            upper_band.iloc[i] = upper_basic[i]
        else:
            upper_band.iloc[i] = upper_band[i-1]
            
        if lower_basic[i] > lower_band[i-1] or df_1d['close'].iloc[i-1] < lower_band[i-1]:
            lower_band.iloc[i] = lower_basic[i]
        else:
            lower_band.iloc[i] = lower_band[i-1]
    
    # Supertrend
    supertrend = pd.Series(index=df_1d.index, dtype=float)
    for i in range(len(supertrend)):
        if i == 0:
            supertrend.iloc[i] = 0.0  # 1 for uptrend, -1 for downtrend
        elif supertrend.iloc[i-1] == -1:
            supertrend.iloc[i] = -1 if df_1d['close'].iloc[i] > upper_band.iloc[i] else 1
        else:
            supertrend.iloc[i] = 1 if df_1d['close'].iloc[i] < lower_band.iloc[i] else -1
    
    # Convert to 1 for uptrend, -1 for downtrend
    supertrend_values = supertrend.values
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend_values)
    
    # Calculate 6h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(10, 20)  # Need 10 for ATR, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(supertrend_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        supertrend_val = supertrend_aligned[i]
        
        # Volume spike: current volume > 1.8 * 20-period average volume
        volume_spike = curr_volume > 1.8 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        broke_above_h3 = curr_close > h3_level
        broke_below_l3 = curr_close < l3_level
        
        # Trend alignment conditions (Supertrend: 1=uptrend, -1=downtrend)
        uptrend = supertrend_val == 1
        downtrend = supertrend_val == -1
        
        # Exit conditions: opposite Camarilla break
        if position != 0:
            # Exit long: close breaks below L3
            if position == 1:
                if curr_close < l3_level:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above H3
            elif position == -1:
                if curr_close > h3_level:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume filters
        if position == 0:
            # Long: break above H3 AND uptrend AND volume spike
            long_condition = broke_above_h3 and uptrend and volume_spike
            
            # Short: break below L3 AND downtrend AND volume spike
            short_condition = broke_below_l3 and downtrend and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dSupertrend10_3_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0