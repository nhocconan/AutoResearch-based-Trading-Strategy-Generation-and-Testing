#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h/1w Camarilla pivot breakout with volume confirmation and ADX trend filter.
# Uses 1w Camarilla pivot levels (calculated from weekly high/low/close) as institutional support/resistance.
# 12h ADX > 25 filters for trending markets to avoid false breakouts in ranging conditions.
# Volume > 1.5x 20-period average confirms institutional participation.
# Works in bull/bear markets as pivot levels adapt to price action and ADX avoids chop.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous weekly bar
    # Formula: 
    # H4 = Close + 1.5*(High - Low)
    # L4 = Close - 1.5*(High - Low)
    # H3 = Close + 1.125*(High - Low)
    # L3 = Close - 1.125*(High - Low)
    # H2 = Close + 0.75*(High - Low)
    # L2 = Close - 0.75*(High - Low)
    # H1 = Close + 0.5*(High - Low)
    # L1 = Close - 0.5*(High - Low)
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    # Calculate levels for each weekly bar
    camarilla_h4 = wc + 1.5 * (wh - wl)
    camarilla_l4 = wc - 1.5 * (wh - wl)
    camarilla_h3 = wc + 1.125 * (wh - wl)
    camarilla_l3 = wc - 1.125 * (wh - wl)
    camarilla_h2 = wc + 0.75 * (wh - wl)
    camarilla_l2 = wc - 0.75 * (wh - wl)
    camarilla_h1 = wc + 0.5 * (wh - wl)
    camarilla_l1 = wc - 0.5 * (wh - wl)
    
    # Align Camarilla levels to 12h timeframe (wait for weekly bar to close)
    h4_12h = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    h3_12h = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    h2_12h = align_htf_to_ltf(prices, df_1w, camarilla_h2)
    l2_12h = align_htf_to_ltf(prices, df_1w, camarilla_l2)
    h1_12h = align_htf_to_ltf(prices, df_1w, camarilla_h1)
    l1_12h = align_htf_to_ltf(prices, df_1w, camarilla_l1)
    
    # Load 12h data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate ADX (14 periods) on 12h data
    # True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = df_12h['high'] - df_12h['high'].shift(1)
    down_move = df_12h['low'].shift(1) - df_12h['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    
    # ADX calculation
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # Align ADX to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_12h, adx_values)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 30)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or
            np.isnan(adx_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX trend filter: only trade when ADX > 25 (trending market)
        trending = adx_12h[i] > 25
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above H3 or H4 with volume and trend
            if (trending and volume_confirmed and
                (close[i] > h3_12h[i] or close[i] > h4_12h[i])):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below L3 or L4 with volume and trend
            elif (trending and volume_confirmed and
                  (close[i] < l3_12h[i] or close[i] < l4_12h[i])):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to H1 or L1 level
            if close[i] < h1_12h[i] or close[i] > l1_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to L1 or H1 level
            if close[i] > l1_12h[i] or close[i] < h1_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Camarilla_Breakout_Volume_ADX_v1"
timeframe = "12h"
leverage = 1.0