#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Camarilla pivot calculation (based on daily OHLC), EMA34 trend, and volume average.
- Camarilla levels: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2.
- Entry: Long when price breaks above H3 with volume > 2.0 * 20-period average volume AND price > 1d EMA34.
         Short when price breaks below L3 with volume > 2.0 * 20-period average volume AND price < 1d EMA34.
- Exit: Opposite Camarilla breakout (price < H3 for long exit, price > L3 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla H3/L3 represent strong intraday support/resistance; breaks with volume indicate institutional participation.
- Works in bull markets (long H3 breaks) and bear markets (short L3 breaks) with EMA34 filter avoiding counter-trend trades.
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
    
    # Calculate Camarilla levels: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3 = df_1d['close'].values + 1.1 * (df_1d['high'].values - df_1d['low'].values) / 2
    camarilla_l3 = df_1d['close'].values - 1.1 * (df_1d['high'].values - df_1d['low'].values) / 2
    
    # Align Camarilla levels and EMA34 to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need 20 for volume MA, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla breakout
        if position != 0:
            # Exit long: price breaks below H3 (bullish breakout failed)
            if position == 1:
                if curr_close < camarilla_h3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above L3 (bearish breakout failed)
            elif position == -1:
                if curr_close > camarilla_l3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend filter and volume spike
        if position == 0:
            # Volume spike: current volume > 2.0 * 20-period average volume
            volume_spike = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            # Long: price breaks above H3 AND volume spike AND price > 1d EMA34
            long_condition = (curr_close > camarilla_h3_aligned[i] and 
                            volume_spike and 
                            curr_close > ema34_1d_aligned[i])
            
            # Short: price breaks below L3 AND volume spike AND price < 1d EMA34
            short_condition = (curr_close < camarilla_l3_aligned[i] and 
                             volume_spike and 
                             curr_close < ema34_1d_aligned[i])
            
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

name = "6h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0