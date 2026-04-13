#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ATR regime filter.
    # Camarilla levels from 1d provide institutional support/resistance.
    # Breakouts with volume + ATR expansion capture strong moves in both bull/bear markets.
    # Target: 50-150 total trades over 4 years = 12-37/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and ATR (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d, 1)
    phigh[0] = high_1d[0]  # First bar uses current values
    plow[0] = low_1d[0]
    pclose[0] = close_1d[0]
    
    # Camarilla levels
    range_ = phigh - plow
    camarilla_h4 = pclose + range_ * 1.1 / 2
    camarilla_l4 = pclose - range_ * 1.1 / 2
    camarilla_h3 = pclose + range_ * 1.1 / 4
    camarilla_l3 = pclose - range_ * 1.1 / 4
    camarilla_h2 = pclose + range_ * 1.1 / 6
    camarilla_l2 = pclose - range_ * 1.1 / 6
    camarilla_h1 = pclose + range_ * 1.1 / 12
    camarilla_l1 = pclose - range_ * 1.1 / 12
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = phigh - plow
    tr2 = np.abs(phigh - pclose)
    tr3 = np.abs(plow - pclose)
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume MA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # Calculate 12h price position for entry triggers
    high_12h = get_htf_data(prices, '12h')['high'].values
    low_12h = get_htf_data(prices, '12h')['low'].values
    high_ma_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    high_ma_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), high_ma_20)
    low_ma_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), low_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 20-period mean (expanding volatility)
        atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
        atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
        volatility_filter = atr_aligned[i] > atr_ma_aligned[i]
        
        # Volume filter: current volume > 20-period MA
        volume_filter = volume[i] > volume_ma_aligned[i]
        
        # Camarilla breakout conditions (using H4/L4 as primary levels)
        breakout_long = close[i] > h4_aligned[i]  # Break above H4
        breakout_short = close[i] < l4_aligned[i]  # Break below L4
        
        # Entry conditions: breakout with volatility AND volume filters
        long_entry = breakout_long and volatility_filter and volume_filter
        short_entry = breakout_short and volatility_filter and volume_filter
        
        # Exit conditions: price returns to opposite Camarilla level
        long_exit = close[i] < l4_aligned[i]
        short_exit = close[i] > h4_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "12h_1d_camarilla_atr_volume_v1"
timeframe = "12h"
leverage = 1.0