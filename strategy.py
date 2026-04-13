#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout + 1d ADX trend filter + volume confirmation
    # Long when: price breaks above 12h Camarilla H3 (1d) AND 1d ADX > 25 AND volume > 1.5x 20-bar avg
    # Short when: price breaks below 12h Camarilla L3 (1d) AND 1d ADX > 25 AND volume > 1.5x 20-bar avg
    # Exit when: price crosses 12h Camarilla pivot point (PP)
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years.
    # Camarilla provides structure; 1d ADX filters weak/choppy markets; volume confirms validity.
    # Works in bull (breakouts with strong trend) and bear (only strong trend-aligned breaks).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ADX(14)
    plus_dm = np.diff(high_1d, prepend=high_1d[0])
    minus_dm = np.diff(low_1d, prepend=low_1d[0]) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr3 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr_period = 14
    tr_ma = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values / tr_ma
    minus_di = 100 * pd.Series(minus_dm).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values / tr_ma
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h Camarilla pivots from 1d OHLC
    # Camarilla formulas: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
    # L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low), PP = (high+low+close)/3
    hl_range = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_h3 = close_1d + 1.1 * hl_range
    camarilla_l3 = close_1d - 1.1 * hl_range
    
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > camarilla_h3_aligned[i-1]  # break above previous Camarilla H3
        breakout_down = close[i] < camarilla_l3_aligned[i-1]  # break below previous Camarilla L3
        
        # Entry conditions with trend filter and volume confirmation
        long_entry = breakout_up and (adx_aligned[i] > 25) and volume_confirmed[i] and position != 1
        short_entry = breakout_down and (adx_aligned[i] > 25) and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < camarilla_pp_aligned[i])
        exit_short = (position == -1 and close[i] > camarilla_pp_aligned[i])
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_adx_volume_v1"
timeframe = "12h"
leverage = 1.0