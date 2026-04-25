#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike
Hypothesis: On 12h timeframe, enter long when price breaks above Camarilla H3 level with volume spike (>2.0x 24-bar average) and 1w EMA34 uptrend; enter short when price breaks below L3 level with volume spike and 1w EMA34 downtrend. Camarilla levels provide high-probability reversal/breakout points from prior 1d action; 1w EMA34 ensures alignment with weekly trend; volume confirms institutional participation. Designed for low trade frequency (target: 12-37/year) to minimize fee drag. Works in bull markets via breakouts with trend and in bear markets via failed breaks/reversions near H3/L3 levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot calculation (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior 1d bar (H3, L3)
    # H3 = close + 1.1*(high - low)/4
    # L3 = close - 1.1*(high - low)/4
    camarilla_h3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 4
    camarilla_l3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 4
    
    # Align Camarilla levels to 12h timeframe (already completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    
    # 1w EMA34 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 2.0 * 24-period average (2x daily volume)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start index: need enough for volume MA (24)
    start_idx = 24
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 + volume spike + 1w uptrend
            long_breakout = curr_close > camarilla_h3_aligned[i]
            # Short: price breaks below L3 + volume spike + 1w downtrend
            short_breakout = curr_close < camarilla_l3_aligned[i]
            
            # Trend filter: price must be on correct side of 1w EMA34
            long_trend = curr_close > ema_34_1w_aligned[i]
            short_trend = curr_close < ema_34_1w_aligned[i]
            
            long_entry = long_breakout and volume_spike[i] and long_trend
            short_entry = short_breakout and volume_spike[i] and short_trend
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below H3 OR trend reverses
            if curr_close < camarilla_h3_aligned[i] or curr_close < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above L3 OR trend reverses
            if curr_close > camarilla_l3_aligned[i] or curr_close > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0