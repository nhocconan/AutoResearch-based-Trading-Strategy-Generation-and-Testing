#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1w EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA34 trend filter to capture major trend direction.
- Camarilla levels: Calculated from prior 1d session (H3/L3 for breakout, H4/L4 for stop).
- Entry: Long when price breaks above H3 with close > H3 AND price > 1w EMA34 AND volume > 1.5 * 20-period average volume.
         Short when price breaks below L3 with close < L3 AND price < 1w EMA34 AND volume > 1.5 * 20-period average volume.
- Exit: Close below L3 (for long) OR close above H3 (for short) OR price crosses 1w EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla H3/L3 breakouts capture institutional order flow clusters effective in both trending and ranging markets.
- 1w EMA34 provides strong long-term trend filter to avoid counter-trend trades during major moves.
- Volume confirmation ensures breakouts have participation, reducing false signals.
- Estimated trades: ~100 total over 4 years (~25/year) based on Camarilla breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_1w = ema(df_1w['close'].values, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w, additional_delay_bars=1)
    
    # Calculate 1w volume average for confirmation
    if len(df_1w) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1w = df_1w['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w, additional_delay_bars=1)
    
    # Calculate daily Camarilla levels (H3, L3, H4, L4) from prior day
    # We need to align daily data to 6h timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    h3 = prev_close + rang * 1.1 / 4
    l3 = prev_close - rang * 1.1 / 4
    h4 = prev_close + rang * 1.1 / 2
    l4 = prev_close - rang * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3, additional_delay_bars=1)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3, additional_delay_bars=1)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4, additional_delay_bars=1)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for 1w EMA34 and daily alignment
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Get current Camarilla levels
        curr_h3 = h3_aligned[i]
        curr_l3 = l3_aligned[i]
        curr_h4 = h4_aligned[i]
        curr_l4 = l4_aligned[i]
        
        # Exit conditions
        if position != 0:
            # Exit long: close below L3 OR price crosses below 1w EMA34
            if position == 1:
                if curr_close < curr_l3 or curr_close < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close above H3 OR price crosses above 1w EMA34
            elif position == -1:
                if curr_close > curr_h3 or curr_close > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend filter and volume confirmation
        if position == 0:
            # Get volume Ma for current bar (from 1w data aligned)
            # Use the aligned volume ratio - if > 1.5 then volume confirmation
            vol_confirmed = vol_ratio_1w_aligned[i] > 1.5 if not np.isnan(vol_ratio_1w_aligned[i]) else False
            
            # Long: Break above H3 with close > H3 AND price > 1w EMA34 AND volume confirmation
            if curr_close > curr_h3 and curr_close > ema34_1w_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Break below L3 with close < L3 AND price < 1w EMA34 AND volume confirmation
            elif curr_close < curr_l3 and curr_close < ema34_1w_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1wEMA34_TrendFilter_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0