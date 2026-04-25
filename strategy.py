#!/usr/bin/env python3
"""
6h Weekly Pivot Donchian Breakout + Volume Spike
Hypothesis: Weekly pivot levels (PP, R1, S1, R2, S2) act as key support/resistance. 
Breakouts above weekly R1 or below S1 with Donchian(20) confirmation and volume spike 
indicate institutional participation. Works in bull markets via breakouts with momentum 
and in bear markets via fade at weekly R2/S2 when price is extended from weekly VWAP.
Target: 12-25 trades/year on 6h (50-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation and VWAP (call ONCE before loop)
    df_w = get_htf_data(prices, '1w')
    
    if len(df_w) < 5:
        return np.zeros(n)
    
    # Calculate weekly VWAP
    typical_price_w = (df_w['high'] + df_w['low'] + df_w['close']) / 3
    vwap_w = (typical_price_w * df_w['volume']).cumsum() / df_w['volume'].cumsum()
    vwap_w_values = vwap_w.values
    
    # Calculate weekly pivot points from previous week OHLC
    # PP = (high + low + close) / 3
    # R1 = 2*PP - low, S1 = 2*PP - high
    # R2 = PP + (high - low), S2 = PP - (high - low)
    prev_week_close = df_w['close'].values
    prev_week_high = df_w['high'].values
    prev_week_low = df_w['low'].values
    
    PP = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    R1 = 2 * PP - prev_week_low
    S1 = 2 * PP - prev_week_high
    R2 = PP + (prev_week_high - prev_week_low)
    S2 = PP - (prev_week_high - prev_week_low)
    
    # Align weekly levels to 6h
    PP_aligned = align_htf_to_ltf(prices, df_w, PP)
    R1_aligned = align_htf_to_ltf(prices, df_w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_w, S1)
    R2_aligned = align_htf_to_ltf(prices, df_w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_w, S2)
    vwap_w_aligned = align_htf_to_ltf(prices, df_w, vwap_w_values)
    
    # Calculate Donchian(20) on 6h for breakout confirmation
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian and volume MA
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(PP_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or np.isnan(vwap_w_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        PP_level = PP_aligned[i]
        R1_level = R1_aligned[i]
        S1_level = S1_aligned[i]
        R2_level = R2_aligned[i]
        S2_level = S2_aligned[i]
        vwap_level = vwap_w_aligned[i]
        upper_donchian = highest_high[i]
        lower_donchian = lowest_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R1 AND Donchian breakout AND volume spike AND price > weekly VWAP
            long_entry = (curr_close > R1_level) and (curr_high > upper_donchian) and vol_spike and (curr_close > vwap_level)
            # Short: price breaks below S1 AND Donchian breakdown AND volume spike AND price < weekly VWAP
            short_entry = (curr_close < S1_level) and (curr_low < lower_donchian) and vol_spike and (curr_close < vwap_level)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                # Fade at weekly R2/S2 when price is extended from VWAP (mean reversion in ranges)
                # Long: price < S2 AND price < VWAP (oversold)
                fade_long = (curr_close < S2_level) and (curr_close < vwap_level)
                # Short: price > R2 AND price > VWAP (overbought)
                fade_short = (curr_close > R2_level) and (curr_close > vwap_level)
                
                if fade_long:
                    signals[i] = 0.20
                    position = 1
                elif fade_short:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below S1 (reversal) OR price < weekly VWAP (mean reversion signal)
            if (curr_close < S1_level) or (curr_close < vwap_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above R1 (reversal) OR price > weekly VWAP (mean reversion signal)
            if (curr_close > R1_level) or (curr_close > vwap_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_VolumeSpike_VWAPFade"
timeframe = "6h"
leverage = 1.0