#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter to capture daily trend direction.
- Camarilla pivot levels: H3 (resistance) and L3 (support) from previous 1d bar.
- Entry: Long when price breaks above H3 with volume > 1.5 * 20-period average volume AND price > 1d EMA34.
         Short when price breaks below L3 with volume > 1.5 * 20-period average volume AND price < 1d EMA34.
- Exit: Opposite Camarilla break (price crosses back below H3 for long, above L3 for short) OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla H3/L3 represent key intraday resistance/support levels where breakouts often indicate strong momentum.
- 1d EMA34 provides medium-term trend filter to avoid counter-trend trades during corrections.
- Volume confirmation ensures breakouts have participation, reducing false signals in low-liquidity periods.
- Works in both bull and bear markets: trend filter adapts to direction, breakouts capture momentum moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for the given bar.
    Based on previous bar's high, low, close.
    Returns: H3, L3 levels
    """
    range_val = high - low
    if range_val == 0:
        return close, close
    H3 = close + range_val * 1.1 / 4
    L3 = close - range_val * 1.1 / 4
    return H3, L3

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for EMA34 and pivot calculation
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
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # We need to shift the 1d data by 1 bar to get previous day's levels
    df_1d_prev = df_1d.shift(1)
    camarilla_H3, camarilla_L3 = calculate_camarilla(
        df_1d_prev['high'].values,
        df_1d_prev['low'].values,
        df_1d_prev['close'].values
    )
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d_prev, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d_prev, camarilla_L3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 35  # Need sufficient data for 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla break OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: price crosses back below H3 OR price falls below 1d EMA34
            if position == 1:
                if curr_close < camarilla_H3_aligned[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses back above L3 OR price rises above 1d EMA34
            elif position == -1:
                if curr_close > camarilla_L3_aligned[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla break with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            vol_ma_20_current = vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else 0
            volume_confirmed = curr_volume > 1.5 * vol_ma_20_current
            
            # Long: Price breaks above H3 AND price > 1d EMA34 AND volume confirmation
            if curr_high > camarilla_H3_aligned[i] and curr_close > ema34_1d_aligned[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L3 AND price < 1d EMA34 AND volume confirmation
            elif curr_low < camarilla_L3_aligned[i] and curr_close < ema34_1d_aligned[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dEMA34_TrendFilter_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0