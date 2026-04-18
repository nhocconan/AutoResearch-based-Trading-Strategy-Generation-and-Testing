#!/usr/bin/env python3
"""
4h_4H_VWAP_Breakout_Volume_TrendFilter
Hypothesis: Price often respects VWAP as dynamic support/resistance. 
Breakouts above/below VWAP with volume (>1.8x 20-bar mean) and aligned daily trend 
(EMA50 > EMA200) capture trends with low frequency. Works in bull/bear by filtering 
counter-trend trades. Target: <30 trades/year per symbol.
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
    
    # Calculate VWAP (typical price * volume cumulative / volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, np.nan)
    
    # Daily trend filter: EMA50 > EMA200 = bullish, < = bearish
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    if len(close_1d) >= 50:
        ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
        ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    else:
        ema50_1d = np.full_like(close_1d, np.nan)
        ema200_1d = np.full_like(close_1d, np.nan)
    daily_trend = ema50_1d - ema200_1d  # >0 bullish, <0 bearish
    
    # Volume confirmation: volume > 1.8x 20-period mean
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Align daily trend to 4h
    daily_trend_4h = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(vwap[i]) or np.isnan(daily_trend_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > 1.8 * vol_ma[i]
        above_vwap = close[i] > vwap[i]
        below_vwap = close[i] < vwap[i]
        
        if position == 0:
            # Long: break above VWAP with volume in bullish daily trend
            if above_vwap and vol_confirm and (daily_trend_4h[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: break below VWAP with volume in bearish daily trend
            elif below_vwap and vol_confirm and (daily_trend_4h[i] < 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below VWAP OR trend turns bearish
            if below_vwap or (daily_trend_4h[i] < 0):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above VWAP OR trend turns bullish
            if above_vwap or (daily_trend_4h[i] > 0):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_4H_VWAP_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0