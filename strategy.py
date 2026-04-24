#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and volume average.
- Camarilla pivot levels: H3 and L3 act as strong intraday resistance/support; breakouts indicate momentum.
- Entry: Long when price breaks above H3 AND price > 1d EMA34 AND volume > 2.0 * 20-period average volume.
         Short when price breaks below L3 AND price < 1d EMA34 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Camarilla breakout (long exits on L3 break, short exits on H3 break).
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla levels derived from prior 1d range, making them adaptive to volatility.
- 1d EMA34 provides smooth trend filter to avoid counter-trend trades.
- Volume confirmation ensures breakouts have participation.
- Works in bull markets (catch breakouts) and bear markets (fade false breaks via volume/trend filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_ = high - low
    if range_ == 0:
        return close, close, close, close
    h3 = close + range_ * 1.1 / 4
    l3 = close - range_ * 1.1 / 4
    h4 = close + range_ * 1.1 / 2
    l4 = close - range_ * 1.1 / 2
    return h3, l3, h4, l4

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
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
    
    # Calculate Camarilla levels from prior 1d data
    # We need to calculate H3, L3 for each 1d bar, then align to 12h
    h3_vals = np.full(len(df_1d), np.nan)
    l3_vals = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        h = df_1d['high'].iloc[i]
        l = df_1d['low'].iloc[i]
        c = df_1d['close'].iloc[i]
        h3, l3, _, _ = calculate_camarilla(h, l, c)
        h3_vals[i] = h3
        l3_vals[i] = l3
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_vals)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_vals)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(35, 20)  # Need 35 for 1d EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla breakout
        if position != 0:
            # Exit long: price breaks below L3
            if position == 1:
                if curr_low < l3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above H3
            elif position == -1:
                if curr_high > h3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend filter and volume confirmation
        if position == 0:
            # Long: price breaks above H3 AND price > 1d EMA34 AND volume confirmation
            long_breakout = curr_high > h3_aligned[i]
            long_trend = curr_close > ema34_1d_aligned[i]
            # Use 1d volume average aligned to 12h for volume confirmation
            vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
            vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
            long_volume = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            # Short: price breaks below L3 AND price < 1d EMA34 AND volume confirmation
            short_breakout = curr_low < l3_aligned[i]
            short_trend = curr_close < ema34_1d_aligned[i]
            short_volume = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            if long_breakout and long_trend and long_volume:
                signals[i] = 0.25
                position = 1
            elif short_breakout and short_trend and short_volume:
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