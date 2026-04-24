#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter to capture intermediate trend direction.
- Camarilla levels: H3 (resistance 3) and L3 (support 3) from prior 12h session.
- Entry: Long when price breaks above H3 AND price > 1d EMA34 AND volume > 2.0 * 20-period average volume.
         Short when price breaks below L3 AND price < 1d EMA34 AND volume > 2.0 * 20-period average volume.
- Exit: Price re-enters the Camarilla H3-L3 range (mean reversion) OR opposite Camarilla breakout.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla H3/L3 are strong intraday support/resistance levels that often contain price.
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
    """Calculate Camarilla pivot levels for the prior period."""
    range_val = high - low
    if range_val == 0:
        return close, close  # Avoid division by zero
    h3 = close + range_val * 1.1 / 4
    l3 = close - range_val * 1.1 / 4
    return h3, l3

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for 1d EMA34 and Camarilla calculation
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Calculate Camarilla levels from prior 12h bar (i-1)
        if i >= 1:
            prior_high = high[i-1]
            prior_low = low[i-1]
            prior_close = close[i-1]
            H3, L3 = calculate_camarilla(prior_high, prior_low, prior_close)
        else:
            # Not enough prior data
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume (using 1d aligned data)
        # For simplicity, use current volume vs 20-period average of current volume
        if i >= 20:
            vol_ma_20_current = np.mean(volume[i-20:i]) if i >= 20 else volume[i]
            volume_confirmed = curr_volume > 2.0 * vol_ma_20_current
        else:
            volume_confirmed = False
        
        # Exit conditions: price re-enters H3-L3 range OR opposite Camarilla breakout
        if position != 0:
            # Exit long: price falls below H3 (re-enters range) OR breaks below L3 (opposite breakout)
            if position == 1:
                if curr_close < H3 or curr_low < L3:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above L3 (re-enters range) OR breaks above H3 (opposite breakout)
            elif position == -1:
                if curr_close > L3 or curr_high > H3:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend filter and volume confirmation
        if position == 0:
            # Long: price breaks above H3 AND price > 1d EMA34 AND volume confirmation
            if curr_high > H3 and curr_close > ema34_1d_aligned[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND price < 1d EMA34 AND volume confirmation
            elif curr_low < L3 and curr_close < ema34_1d_aligned[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_TrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0