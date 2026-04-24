#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA200 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA200 trend filter to capture major trend direction.
- Camarilla levels: H3 (resistance) and L3 (support) from prior 1d range.
- Entry: Long when price breaks above H3 AND price > 1d EMA200 AND volume > 2.0 * 20-period average volume.
         Short when price breaks below L3 AND price < 1d EMA200 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Camarilla level touch (L3 for long, H3 for short) OR price crosses 1d EMA200 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla H3/L3 breakouts capture institutional order flow around key intraday levels.
- 1d EMA200 provides strong long-term trend filter to avoid counter-trend trades during major moves.
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
    """Calculate Camarilla pivot levels (H3, L3) from prior period."""
    range_val = high - low
    h3 = close + range_val * 1.1 / 4
    l3 = close - range_val * 1.1 / 4
    return h3, l3

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need sufficient data for 1d EMA200
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 205:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    ema200_1d = ema(df_1d['close'].values, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d, additional_delay_bars=1)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d, additional_delay_bars=1)
    
    # Calculate Camarilla levels from prior 1d
    df_1d_for_camarilla = get_htf_data(prices, '1d')
    if len(df_1d_for_camarilla) < 2:
        return np.zeros(n)
    
    camarilla_h3, camarilla_l3 = calculate_camarilla(
        df_1d_for_camarilla['high'].values,
        df_1d_for_camarilla['low'].values,
        df_1d_for_camarilla['close'].values
    )
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_h3, additional_delay_bars=1)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_l3, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 200  # Need sufficient data for 1d EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla level touch OR price crosses 1d EMA200 in opposite direction
        if position != 0:
            # Exit long: price touches L3 OR price falls below 1d EMA200
            if position == 1:
                if curr_close <= camarilla_l3_aligned[i] or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price touches H3 OR price rises above 1d EMA200
            elif position == -1:
                if curr_close >= camarilla_h3_aligned[i] or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 2.0 * 20-period average volume
            # Use aligned volume ratio for consistency
            vol_confirmed = vol_ratio_1d_aligned[i] > 2.0
            
            # Long: Price breaks above H3 AND price > 1d EMA200 AND volume confirmation
            if curr_close > camarilla_h3_aligned[i] and curr_close > ema200_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L3 AND price < 1d EMA200 AND volume confirmation
            elif curr_close < camarilla_l3_aligned[i] and curr_close < ema200_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA200_TrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0