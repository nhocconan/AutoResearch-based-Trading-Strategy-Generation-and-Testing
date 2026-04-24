#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA50 trend filter to capture major trend direction and avoid counter-trend trades.
- Camarilla levels (H3, L3, H4, L4) calculated from prior 1d session to identify key intraday support/resistance.
- Entry: Long when price breaks above H3 with close > H3 AND price > 1w EMA50 AND volume > 1.5 * 20-period average volume.
         Short when price breaks below L3 with close < L3 AND price < 1w EMA50 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Camarilla break (L3 for longs, H3 for shorts) OR price crosses 1w EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla H3/L3 levels act as magnet points; breaks often indicate continuation with institutional participation.
- 1w EMA50 filter ensures alignment with major trend, reducing losses during sideways/choppy periods.
- Volume confirmation ensures breakouts have sufficient participation, reducing false signals.
- Estimated trades: ~80 total over 4 years (~20/year) based on Camarilla break frequency with filters.
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
    camarilla = {
        'H4': close + range_val * 1.1 / 2,
        'H3': close + range_val * 1.1 / 4,
        'L3': close - range_val * 1.1 / 4,
        'L4': close - range_val * 1.1 / 2
    }
    return camarilla

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need sufficient data for 1w EMA50
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate 1w volume average for confirmation
    if len(df_1w) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1w = df_1w['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w, additional_delay_bars=1)
    
    # Calculate 1d Camarilla levels (prior session)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    camarilla_H3 = np.full(n, np.nan)
    camarilla_L3 = np.full(n, np.nan)
    camarilla_H4 = np.full(n, np.nan)
    camarilla_L4 = np.full(n, np.nan)
    
    # Calculate Camarilla levels for each 1d bar and align to 6h
    for i in range(len(df_1d)):
        camarilla_levels = calculate_camarilla(df_1d['high'].iloc[i], df_1d['low'].iloc[i], df_1d['close'].iloc[i])
        camarilla_H3[i] = camarilla_levels['H3']
        camarilla_L3[i] = camarilla_levels['L3']
        camarilla_H4[i] = camarilla_levels['H4']
        camarilla_L4[i] = camarilla_levels['L4']
    
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3, additional_delay_bars=1)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3, additional_delay_bars=1)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4, additional_delay_bars=1)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 200  # Need sufficient data for 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(camarilla_L3_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla break OR price crosses 1w EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below L3 OR price falls below 1w EMA50
            if position == 1:
                if curr_low < camarilla_L3_aligned[i] or curr_close < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above H3 OR price rises above 1w EMA50
            elif position == -1:
                if curr_high > camarilla_H3_aligned[i] or curr_close > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla break with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            vol_ma_20_idx = min(i, len(vol_ma_20)-1) if len(vol_ma_20) > 0 else 0
            volume_confirmed = curr_volume > 1.5 * vol_ma_20[vol_ma_20_idx] if len(vol_ma_20) > 0 else False
            
            # Long: Price breaks above H3 with close > H3 AND price > 1w EMA50 AND volume confirmation
            if curr_close > camarilla_H3_aligned[i] and curr_close > ema50_1w_aligned[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L3 with close < L3 AND price < 1w EMA50 AND volume confirmation
            elif curr_close < camarilla_L3_aligned[i] and curr_close < ema50_1w_aligned[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1wEMA50_TrendFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0