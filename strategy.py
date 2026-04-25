#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA21 trend filter and volume spike filter.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA21 trend direction and Camarilla pivot levels (H3/L3) from prior 4h bar.
- Camarilla Pivots: H3, L3 levels from prior 4h OHLC for breakout logic.
- Trend Filter: 4h EMA21 must align with breakout direction (long: close > EMA21, short: close < EMA21).
- Volume Filter: Current 1h volume > 1.8 * 20-period average 1h volume to confirm strong momentum.
- Session Filter: Trade only between 08:00-20:00 UTC to avoid low-liquidity periods.
- Entry: Long when close > H3 AND close > 4h EMA21 AND volume spike AND in session.
         Short when close < L3 AND close < 4h EMA21 AND volume spike AND in session.
- Exit: Opposite Camarilla break (long exits when close < L3, short exits when close > H3).
- Signal size: 0.20 discrete to minimize fee drag.
- Designed to capture strong momentum bursts aligned with 4h trend while filtering chop/whipsaws.
- Works in bull markets (trend continuation) and bear markets (trend continuation down).
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
    
    # Calculate 4h EMA21 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 4h Camarilla pivots (H3, L3) from prior 4h bar OHLC
    prev_high = df_4h['high'].shift(1).values  # Shifted to avoid look-ahead
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Camarilla H3 and L3 levels (using standard Camarilla formula)
    camarilla_range = prev_high - prev_low
    h3 = prev_close + camarilla_range * 1.1 / 2
    l3 = prev_close - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (waits for 4h bar close)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    
    # Calculate 1h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(21, 20)  # Need 21 for EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_21_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08:00-20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        curr_close = close[i]
        curr_volume = volume[i]
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        ema_21_level = ema_21_4h_aligned[i]
        
        # Volume spike: current volume > 1.8 * 20-period average volume
        volume_spike = curr_volume > 1.8 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        broke_above_h3 = curr_close > h3_level
        broke_below_l3 = curr_close < l3_level
        
        # Trend alignment conditions
        above_ema = curr_close > ema_21_level
        below_ema = curr_close < ema_21_level
        
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
        
        # Entry conditions: Camarilla breakout with trend, volume, and session filters
        if position == 0 and in_session:
            # Long: break above H3 AND above EMA21 AND volume spike
            long_condition = broke_above_h3 and above_ema and volume_spike
            
            # Short: break below L3 AND below EMA21 AND volume spike
            short_condition = broke_below_l3 and below_ema and volume_spike
            
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

name = "1h_Camarilla_H3L3_Breakout_4hEMA21_Trend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0