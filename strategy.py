#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_Trend_v1
Hypothesis: On 12h timeframe, buy breakouts above daily Camarilla H3 with daily price above 200 EMA (bullish trend),
sell breakdowns below daily Camarilla L3 with daily price below 200 EMA (bearish trend). Exit at opposite H4/L4 levels.
Uses daily volatility regime filter to avoid choppy markets. Designed for low trade frequency
(15-30/year) by requiring multiple confluence factors. Works in bull/bear via daily trend filter
and mean-reversion exit at Camarilla levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Breakout_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need enough for EMA200
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === DAILY CAMARILLA LEVELS (using previous day) ===
    # Shift by 1 to avoid look-ahead: use previous day's OHLC
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    range_prev = high_prev - low_prev
    
    h5 = close_prev + (range_prev * 1.1 / 2)
    h4 = close_prev + (range_prev * 1.1)
    h3 = close_prev + (range_prev * 1.1 / 4)
    l3 = close_prev - (range_prev * 1.1 / 4)
    l4 = close_prev - (range_prev * 1.1)
    l5 = close_prev - (range_prev * 1.1 / 2)
    
    # === DAILY 200 EMA FOR TREND ===
    close_series = pd.Series(close_1d)
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    # Trend: price above/below EMA200
    trend_up = close_1d > ema_200
    trend_down = close_1d < ema_200
    
    # === DAILY VOLATILITY REGIME (ATR-based) ===
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # ATR(14)
    atr_14 = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr_14[i] = np.nan
        elif i == 14:
            atr_14[i] = np.nanmean(tr[1:i+1])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    # ATR MA(30) for regime
    atr_ma = np.zeros_like(atr_14)
    for i in range(len(atr_14)):
        if i < 30:
            atr_ma[i] = np.nan
        else:
            atr_ma[i] = np.mean(atr_14[i-29:i+1])
    # Low volatility regime (trending market)
    vol_regime = atr_14 < atr_ma
    
    # Align all daily data to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down.astype(float))
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Only trade in low volatility (trending) regime
        in_trend_regime = vol_regime_aligned[i] > 0.5
        
        # Entry conditions: price breaks H3/L3 with daily trend confirmation
        long_setup = (close[i] > h3_aligned[i]) and trend_up_aligned[i] > 0.5 and in_trend_regime
        short_setup = (close[i] < l3_aligned[i]) and trend_down_aligned[i] > 0.5 and in_trend_regime
        
        # Exit conditions: mean reversion to opposite H4/L4 levels
        exit_long = close[i] < l4_aligned[i]
        exit_short = close[i] > h4_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals