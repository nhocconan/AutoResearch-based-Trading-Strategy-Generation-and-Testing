#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h/1d Camarilla pivot breakout with volume confirmation and session filter
    # Uses 4h trend direction + 1d Camarilla levels for structure + volume spike for confirmation
    # Target: 15-37 trades/year per symbol on 1h timeframe
    # Works in bull/bear by fading false breakouts in ranging markets and catching real breakouts in trends
    
    # Session filter: 8:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA(34) for trend
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d
    # Camarilla: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # H2 = close + 1.1*(high-low)/6, L2 = close - 1.1*(high-low)/6
    # H1 = close + 1.1*(high-low)/12, L1 = close - 1.1*(high-low)/12
    camarilla_h4 = np.zeros(len(df_1d))
    camarilla_l4 = np.zeros(len(df_1d))
    camarilla_h3 = np.zeros(len(df_1d))
    camarilla_l3 = np.zeros(len(df_1d))
    camarilla_h2 = np.zeros(len(df_1d))
    camarilla_l2 = np.zeros(len(df_1d))
    camarilla_h1 = np.zeros(len(df_1d))
    camarilla_l1 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_h4[i] = camarilla_l4[i] = camarilla_h3[i] = camarilla_l3[i] = camarilla_h2[i] = camarilla_l2[i] = camarilla_h1[i] = camarilla_l1[i] = np.nan
        else:
            diff = high_1d[i-1] - low_1d[i-1]
            camarilla_h4[i] = close_1d[i-1] + 1.1 * diff / 2
            camarilla_l4[i] = close_1d[i-1] - 1.1 * diff / 2
            camarilla_h3[i] = close_1d[i-1] + 1.1 * diff / 4
            camarilla_l3[i] = close_1d[i-1] - 1.1 * diff / 4
            camarilla_h2[i] = close_1d[i-1] + 1.1 * diff / 6
            camarilla_l2[i] = close_1d[i-1] - 1.1 * diff / 6
            camarilla_h1[i] = close_1d[i-1] + 1.1 * diff / 12
            camarilla_l1[i] = close_1d[i-1] - 1.1 * diff / 12
    
    # Align Camarilla levels to 1h timeframe
    h4_1h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_1h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_1h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_1h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_1h = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_1h = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_1h = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_1h = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Volume spike filter: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(h4_1h[i]) or np.isnan(l4_1h[i]) or
            np.isnan(h3_1h[i]) or np.isnan(l3_1h[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 4h EMA
        uptrend = close[i] > ema_34_4h_aligned[i]
        downtrend = close[i] < ema_34_4h_aligned[i]
        
        # Camarilla breakout conditions
        breakout_h4 = close[i] > h4_1h[i]
        breakdown_l4 = close[i] < l4_1h[i]
        breakout_h3 = close[i] > h3_1h[i]
        breakdown_l3 = close[i] < l3_1h[i]
        
        # Entry conditions: fade extreme levels in ranging markets, break intermediate levels in trends
        # In ranging markets (price near H4/L4), fade the extreme
        # In trending markets, break H3/L3 for continuation
        long_entry = (breakout_h3 and uptrend and volume_filter[i]) or \
                     (close[i] < l4_1h[i] and close[i] > l3_1h[i] and not uptrend and not downtrend and volume_filter[i])
        short_entry = (breakdown_l3 and downtrend and volume_filter[i]) or \
                      (close[i] > h4_1h[i] and close[i] < h3_1h[i] and not uptrend and not downtrend and volume_filter[i])
        
        # Exit conditions: opposite Camarilla level or volume drying up
        long_exit = (close[i] < l3_1h[i]) or (not volume_filter[i])
        short_exit = (close[i] > h3_1h[i]) or (not volume_filter[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_camarilla_breakout_vol_filter_v1"
timeframe = "1h"
leverage = 1.0