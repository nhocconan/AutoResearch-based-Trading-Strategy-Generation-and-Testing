#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend filter to capture intermediate trend direction.
- Camarilla levels: H3 (resistance) and L3 (support) from prior 1d session.
- Entry: Long when price breaks above H3 with volume > 1.5 * 20-period average AND price > 1d EMA34.
         Short when price breaks below L3 with volume > 1.5 * 20-period average AND price < 1d EMA34.
- Exit: Opposite Camarilla break (L3 for longs, H3 for shorts) OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla H3/L3 represent key intraday pivot levels where breakouts often occur with follow-through.
- 1d EMA34 provides intermediate trend filter to avoid counter-trend trades during corrections.
- Volume spike confirmation ensures breakouts have participation, reducing false signals.
- Estimated trades: ~100 total over 4 years (~25/year) based on Camarilla breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the session."""
    range_val = high - low
    h3 = close + range_val * 1.1 / 4
    l3 = close - range_val * 1.1 / 4
    return h3, l3

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d, additional_delay_bars=1)
    
    # Calculate Camarilla levels from prior 1d session
    df_1d_for_cam = get_htf_data(prices, '1d')
    if len(df_1d_for_cam) < 2:
        return np.zeros(n)
    
    # Use prior 1d OHLC for Camarilla calculation (shifted by 1 to avoid look-ahead)
    cam_high = df_1d_for_cam['high'].values
    cam_low = df_1d_for_cam['low'].values
    cam_close = df_1d_for_cam['close'].values
    
    h3_levels, l3_levels = calculate_camarilla(cam_high, cam_low, cam_close)
    h3_aligned = align_htf_to_ltf(prices, df_1d_for_cam, h3_levels, additional_delay_bars=1)
    l3_aligned = align_htf_to_ltf(prices, df_1d_for_cam, l3_levels, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla break OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below L3 OR price falls below 1d EMA34
            if position == 1:
                if curr_close < l3_aligned[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above H3 OR price rises above 1d EMA34
            elif position == -1:
                if curr_close > h3_aligned[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla break with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 1.5 * 20-period average
            vol_confirmed = curr_volume > 1.5 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False
            
            # Long: Price breaks above H3 AND price > 1d EMA34 AND volume confirmation
            if curr_close > h3_aligned[i] and curr_close > ema34_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L3 AND price < 1d EMA34 AND volume confirmation
            elif curr_close < l3_aligned[i] and curr_close < ema34_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_TrendFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0