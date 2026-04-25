#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeRegime
Hypothesis: Camarilla H3/L3 breakout with 1d EMA34 trend filter, volume confirmation, and choppiness regime filter.
Designed for 19-50 trades/year (75-200 over 4 years). Works in bull markets via breakout continuation
and bear markets via trend following. The choppiness filter avoids ranging markets where breakouts fail.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation and EMA34 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Prior 1d bar OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H3, L3 (standard breakout levels)
    camarilla_range = prev_high - prev_low
    h3 = prev_close + camarilla_range * 1.1 / 4
    l3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness regime filter: CHOP > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor breakouts)
    # Calculate using 14-period high/low range vs true range
    hl_range = high - low
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    sum_hL14 = pd.Series(hl_range).rolling(window=14, min_periods=14).sum().values
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr14 / sum_hL14) / np.log10(14)
    chop_regime = chop < 61.8  # True when trending (CHOP < 61.8), False when ranging
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA (34), volume MA (20), chop (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H3/L3 breakout + volume spike + 1d EMA34 trend alignment + trending regime
            long_breakout = curr_high > h3_aligned[i]
            short_breakout = curr_low < l3_aligned[i]
            
            # Trend filter: price must be on correct side of 1d EMA34
            long_trend = curr_close > ema_34_1d_aligned[i]
            short_trend = curr_close < ema_34_1d_aligned[i]
            
            long_entry = (long_breakout and volume_spike[i] and long_trend and chop_regime[i])
            short_entry = (short_breakout and volume_spike[i] and short_trend and chop_regime[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below Camarilla H3 (failed breakout) or trend reverses
            if curr_close < h3_aligned[i] or curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Camarilla L3 (failed breakout) or trend reverses
            if curr_close > l3_aligned[i] or curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeRegime"
timeframe = "4h"
leverage = 1.0