#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter to capture intermediate trend direction.
- Camarilla levels: Calculate H3/L3 from previous 1d candle (loaded via get_htf_data).
- Entry: Long when price breaks above H3 with volume > 2.0 * 20-period average volume AND price > 1d EMA34.
         Short when price breaks below L3 with volume > 2.0 * 20-period average volume AND price < 1d EMA34.
- Exit: Opposite Camarilla break (price < L3 for long, price > H3 for short) OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla H3/L3 represents natural resistance/support levels where breakouts often indicate strong momentum.
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
    """Calculate Camarilla pivot levels (H3, L3) from previous day's OHLC."""
    range_val = high - low
    h3 = close + range_val * 1.1 / 4
    l3 = close - range_val * 1.1 / 4
    return h3, l3

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for indicators
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
    
    # Calculate Camarilla levels from previous 1d candle
    # We need to shift the 1d data by 1 to get previous day's levels
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Calculate H3/L3 for previous 1d candle
    camarilla_h3_1d, camarilla_l3_1d = calculate_camarilla(prev_high_1d, prev_low_1d, prev_close_1d)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d, additional_delay_bars=1)
    camarilla_l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 35  # Need sufficient data for 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(camarilla_h3_1d_aligned[i]) or np.isnan(camarilla_l3_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Get current Camarilla levels (from previous 1d candle)
        curr_h3 = camarilla_h3_1d_aligned[i]
        curr_l3 = camarilla_l3_1d_aligned[i]
        
        # Get 1d EMA34 value
        curr_ema34 = ema34_1d_aligned[i]
        
        # Get 1d volume ratio (current 1d bar's volume vs 20-period MA)
        # Note: vol_ratio_1d_aligned gives us the ratio for the completed 1d bar
        curr_vol_ratio = vol_ratio_1d_aligned[i]
        
        # Exit conditions: opposite Camarilla break OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below L3 OR price falls below 1d EMA34
            if position == 1:
                if curr_low < curr_l3 or curr_close < curr_ema34:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above H3 OR price rises above 1d EMA34
            elif position == -1:
                if curr_high > curr_h3 or curr_close > curr_ema34:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla break with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current 1d bar's volume > 2.0 * 20-period average
            volume_confirmed = curr_vol_ratio > 2.0
            
            # Long: Price breaks above H3 AND price > 1d EMA34 AND volume confirmation
            if curr_high > curr_h3 and curr_close > curr_ema34 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L3 AND price < 1d EMA34 AND volume confirmation
            elif curr_low < curr_l3 and curr_close < curr_ema34 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dEMA34_TrendFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0