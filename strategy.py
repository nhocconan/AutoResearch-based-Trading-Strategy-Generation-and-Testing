#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA34 trend filter to ensure alignment with intermediate trend.
- Camarilla levels: H3/L3 from prior day provide institutional pivot points for breakout/mean reversion.
- Entry: Long when price breaks above Camarilla H3 AND price > 12h EMA34 AND volume > 2.0 * 20-period average volume.
         Short when price breaks below Camarilla L3 AND price < 12h EMA34 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Camarilla breakout (long exits on L3 break, short exits on H3 break).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla H3/L3 breakouts capture institutional order flow; EMA34 filter avoids counter-trend trades.
- Volume confirmation ensures breakouts have participation, reducing false signals.
- Works in bull markets (trend continuation) and bear markets (mean reversion off institutional levels).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period."""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    camarilla_close = close
    h3 = camarilla_close + range_val * 1.1 / 4
    l3 = camarilla_close - range_val * 1.1 / 4
    h4 = camarilla_close + range_val * 1.1 / 2
    l4 = camarilla_close - range_val * 1.1 / 2
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
    
    # Calculate 12h trend filter: EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_12h = ema(df_12h['close'].values, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 12h volume average for confirmation
    if len(df_12h) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = df_12h['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # Calculate Camarilla levels from daily data (prior day's H/L/C)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use prior day's OHLC for today's Camarilla levels
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        h3, l3, _, _ = calculate_camarilla(
            df_1d['high'].values[i-1],
            df_1d['low'].values[i-1],
            df_1d['close'].values[i-1]
        )
        camarilla_h3[i] = h3
        camarilla_l3[i] = l3
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(35, 1)  # Need 35 for 12h EMA34, 1 for Camarilla (using prior day)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
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
            # Exit long: price breaks below Camarilla L3
            if position == 1:
                if curr_low < camarilla_l3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3
            elif position == -1:
                if curr_high > camarilla_h3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend filter and volume confirmation
        if position == 0:
            # Long: price breaks above Camarilla H3 AND price > 12h EMA34 AND volume confirmation
            long_breakout = curr_high > camarilla_h3_aligned[i]
            long_trend = curr_close > ema34_12h_aligned[i]
            long_volume = curr_volume > 2.0 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False
            
            # Short: price breaks below Camarilla L3 AND price < 12h EMA34 AND volume confirmation
            short_breakout = curr_low < camarilla_l3_aligned[i]
            short_trend = curr_close < ema34_12h_aligned[i]
            short_volume = curr_volume > 2.0 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False
            
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

name = "4h_Camarilla_H3L3_Breakout_12hEMA34_TrendFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0