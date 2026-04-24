#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Camarilla pivot calculation (based on daily high/low/close) and EMA34 trend.
- Camarilla: H3 = close + 1.1/12*(high-low), L3 = close - 1.1/12*(high-low).
- Entry: Long when price crosses above H3 AND price > 1d EMA34 AND volume > 2.0 * 20-period average volume.
         Short when price crosses below L3 AND price < 1d EMA34 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Camarilla level touch (price touches L3 for long exit, H3 for short exit) OR EMA34 trend reversal.
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla levels act as intraday support/resistance; breaks indicate strong momentum.
- Works in bull markets (long breaks above H3) and bear markets (short breaks below L3) with trend filter.
- Volume spike confirms institutional participation, reducing false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (H3, L3) and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for EMA34 and volume MA
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = ema(df_1d['close'].values, 34)
    
    # Calculate Camarilla levels: H3 = close + 1.1/12*(high-low), L3 = close - 1.1/12*(high-low)
    camarilla_multiplier = 1.1 / 12
    daily_range = df_1d['high'].values - df_1d['low'].values
    h3_1d = df_1d['close'].values + camarilla_multiplier * daily_range
    l3_1d = df_1d['close'].values - camarilla_multiplier * daily_range
    
    # Align Camarilla levels and EMA34 to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need 20 for volume MA, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Calculate price crosses: above H3 or below L3
        if i > 0:
            prev_close = close[i-1]
            cross_above_h3 = prev_close <= h3_aligned[i-1] and curr_close > h3_aligned[i]
            cross_below_l3 = prev_close >= l3_aligned[i-1] and curr_close < l3_aligned[i]
        else:
            cross_above_h3 = False
            cross_below_l3 = False
        
        # Exit conditions: opposite Camarilla level touch OR EMA34 trend reversal
        if position != 0:
            # Exit long: price touches L3 (support) OR price < EMA34 (trend reversal)
            if position == 1:
                if curr_low <= l3_aligned[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price touches H3 (resistance) OR price > EMA34 (trend reversal)
            elif position == -1:
                if curr_high >= h3_aligned[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend filter and volume spike
        if position == 0:
            # Volume spike: current volume > 2.0 * 20-period average volume
            volume_spike = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            # Long: price crosses above H3 AND price > EMA34 AND volume spike
            long_condition = (cross_above_h3 and 
                            curr_close > ema34_1d_aligned[i] and
                            volume_spike)
            
            # Short: price crosses below L3 AND price < EMA34 AND volume spike
            short_condition = (cross_below_l3 and 
                             curr_close < ema34_1d_aligned[i] and
                             volume_spike)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0