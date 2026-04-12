#!/usr/bin/env python3
"""
1d_1w_Camarilla_Breakout_Trend
Hypothesis: Use weekly market regime to filter daily Camarilla breakouts.
In trending weeks (price > weekly VWAP), trade breakouts in direction of trend.
In ranging weeks (price near weekly VWAP), fade extremes at H3/L3.
This avoids counter-trend breakouts in strong trends and whipsaws in ranges.
Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Breakout_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY REGIME FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly VWAP approximation: (H+L+C)/3
    typical_price_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3
    vwap_1w = typical_price_1w  # Simplified VWAP
    weekly_close = df_1w['close'].values
    
    # Trend: price above/below VWAP
    trend_up = weekly_close > vwap_1w
    trend_down = weekly_close < vwap_1w
    ranging = np.abs(weekly_close - vwap_1w) / vwap_1w < 0.02  # Within 2% of VWAP
    
    # Align to daily
    trend_up_1d = align_htf_to_ltf(prices, df_1w, trend_up)
    trend_down_1d = align_htf_to_ltf(prices, df_1w, trend_down)
    ranging_1d = align_htf_to_ltf(prices, df_1w, ranging)
    
    # === DAILY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    h3_1d = camarilla_h3  # Already daily
    l3_1d = camarilla_l3
    h4_1d = camarilla_h4
    l4_1d = camarilla_l4
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h3_1d[i]) or np.isnan(l3_1d[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(trend_up_1d[i]) or np.isnan(trend_down_1d[i]) or np.isnan(ranging_1d[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # In trending weeks: trade breakouts in direction of trend
        if trend_up_1d[i] and not ranging_1d[i]:
            # Long: break above H3 in uptrend week
            if close[i] > h3_1d[i] * 1.001 and vol_ratio[i] > 1.5:
                if position != 1:
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Exit: close below L3 or reverse signal
            elif position == 1 and (close[i] < l3_1d[i] * 0.999 or trend_down_1d[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else 0.0
                
        elif trend_down_1d[i] and not ranging_1d[i]:
            # Short: break below L3 in downtrend week
            if close[i] < l3_1d[i] * 0.999 and vol_ratio[i] > 1.5:
                if position != -1:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            # Exit: close above H3 or reverse signal
            elif position == -1 and (close[i] > h3_1d[i] * 1.001 or trend_up_1d[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25 if position == -1 else 0.0
                
        else:
            # In ranging weeks: fade extremes at H3/L3
            # Long: pullback to L3 with rejection
            if close[i] < l3_1d[i] * 1.002 and close[i] > l3_1d[i] * 0.998:
                if position != 1:
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: pullback to H3 with rejection
            elif close[i] > h3_1d[i] * 0.998 and close[i] < h3_1d[i] * 1.002:
                if position != -1:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            # Exit: return to VWAP area or opposite extreme
            elif position == 1 and close[i] > h3_1d[i] * 0.998:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] < l3_1d[i] * 1.002:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals