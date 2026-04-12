#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_Plus_TIPS_v2
Hypothesis: Use daily Camarilla H3/L3 breakouts with 12h EMA trend filter and volume spike confirmation.
Adds TIPS (Treasury Inflation-Protected Securities) as a macro regime filter to improve performance in both bull and bear markets.
TIPS data is sourced from the 'tips' macro dataset, representing real yields. Rising real yields (TIPS up) often correlate with risk-off environments, while falling real yields support risk assets.
In bull markets (falling/rising TIPS with price strength), we allow longs. In bear markets (rising TIPS with price weakness), we favor shorts.
This macro filter should reduce whipsaws and improve trend fidelity.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from macro_data import get_macro_data, align_macro_to_ltf

name = "12h_1d_Camarilla_Breakout_Plus_TIPS_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (H3, L3, H4, L4)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        range_val = high_1d[i] - low_1d[i]
        camarilla_h3[i] = close_1d[i] + range_val * 1.1 / 6
        camarilla_l3[i] = close_1d[i] - range_val * 1.1 / 6
        camarilla_h4[i] = close_1d[i] + range_val * 1.1 / 4
        camarilla_l4[i] = close_1d[i] - range_val * 1.1 / 4
    
    # Align to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === 12-HOUR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_12h = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # === TIPS MACRO FILTER ===
    # TIPS (Treasury Inflation-Protected Securities) - real yields
    # Rising real yields = risk-off, Falling real yields = risk-on
    df_tips = get_macro_data(prices, 'tips')
    if len(df_tips) < 2:
        return np.zeros(n)
    
    tips_close = df_tips['close'].values
    # Use 20-period EMA of TIPS to determine trend
    tips_ema20 = pd.Series(tips_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tips_ema20_aligned = align_macro_to_ltf(prices, df_tips, tips_ema20)
    
    # TIPS trend: rising = 1, falling = -1
    tips_trend = np.where(tips_ema20_aligned > np.roll(tips_ema20_aligned, 1), 1, -1)
    # Handle first element
    tips_trend[0] = 0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or np.isnan(ema20_12h_12h[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(tips_ema20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get current 12h close 
        close_12h_arr = df_12h['close'].values
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h_arr)
        trend_up = close_12h_aligned[i] > ema20_12h_12h[i]
        trend_down = close_12h_aligned[i] < ema20_12h_12h[i]
        
        # TIPS filter: allow longs when TIPS falling or flat (risk-on), shorts when TIPS rising (risk-off)
        tips_risk_on = tips_trend[i] <= 0  # TIPS falling or flat = good for risk assets
        tips_risk_off = tips_trend[i] > 0   # TIPS rising = risk-off
        
        # Long: break above H3 in uptrend with volume surge and TIPS risk-on
        long_signal = (trend_up and 
                      close[i] > h3_12h[i] * 1.001 and  # Break above H3
                      vol_ratio[i] > 2.0 and
                      tips_risk_on)
        
        # Short: break below L3 in downtrend with volume surge and TIPS risk-off
        short_signal = (trend_down and 
                       close[i] < l3_12h[i] * 0.999 and  # Break below L3
                       vol_ratio[i] > 2.0 and
                       tips_risk_off)
        
        # Exit: trend reversal or retracement to H4/L4
        exit_long = (position == 1 and 
                    (not trend_up or close[i] <= h4_12h[i]))
        exit_short = (position == -1 and 
                     (not trend_down or close[i] >= l4_12h[i]))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals