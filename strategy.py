#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R mean reversion with 1d ADX regime filter.
    # Williams %R identifies overbought/oversold conditions.
    # 1d ADX > 25 filters for trending markets (avoid chop).
    # Long when %R < -80 and ADX > 25, short when %R > -20 and ADX > 25.
    # Exit when %R crosses -50 (mean reversion target).
    # Target: 50-150 total trades over 4 years = 12-37/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    period = 14
    plus_dm = np.diff(df_1d['high'].values)
    minus_dm = np.diff(df_1d['low'].values) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(np.diff(df_1d['high'].values))
    tr2 = np.abs(np.diff(df_1d['low'].values))
    tr3 = np.abs(np.diff(df_1d['close'].values))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # ADX filter: only trade when trending (ADX > 25)
        adx_filter = adx_aligned[i] > 25
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        exit_long = williams_r[i] > -50
        exit_short = williams_r[i] < -50
        
        # Entry conditions: mean reversion in trending market
        long_entry = oversold and adx_filter
        short_entry = overbought and adx_filter
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_williams_r_adx_regime_v1"
timeframe = "6h"
leverage = 1.0