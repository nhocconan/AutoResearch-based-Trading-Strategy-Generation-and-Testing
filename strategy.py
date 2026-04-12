#!/usr/bin/env python3
"""
6h_12h_1d_Triple_Pullback_Strategy
Hypothesis: Enter long on pullbacks to 12h EMA21 when 1d trend is up (price > 1d EMA50) and 12h momentum is bullish (MACD > 0). Enter short on pullbacks to 12h EMA21 when 1d trend is down (price < 1d EMA50) and 12h momentum is bearish (MACD < 0). Uses 1d trend filter to avoid counter-trend trades, and 12h EMA21 as dynamic support/resistance. Designed for low trade frequency (15-30/year) by requiring trend alignment and pullback entries. Works in bull via long bias in uptrends and in bear via short bias in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_Triple_Pullback_Strategy"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1D TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    if len(close_1d) >= 50:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50_1d = np.full_like(close_1d, np.nan)
    
    # === 12H INDICATORS ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA21 for pullback entries
    if len(close_12h) >= 21:
        ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    else:
        ema_21_12h = np.full_like(close_12h, np.nan)
    
    # 12h MACD for momentum confirmation
    if len(close_12h) >= 34:
        ema_12 = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean().values
        ema_26 = pd.Series(close_12h).ewm(span=26, adjust=False, min_periods=26).mean().values
        macd_line = ema_12 - ema_26
        signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
        macd_hist = macd_line - signal_line
    else:
        macd_line = np.full_like(close_12h, np.nan)
        signal_line = np.full_like(close_12h, np.nan)
        macd_hist = np.full_like(close_12h, np.nan)
    
    # Align all 1d and 12h indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    macd_hist_aligned = align_htf_to_ltf(prices, df_12h, macd_hist)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_21_12h_aligned[i]) or 
            np.isnan(macd_hist_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend alignment: 1d price vs 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Momentum: 12h MACD histogram
        mom_bullish = macd_hist_aligned[i] > 0
        mom_bearish = macd_hist_aligned[i] < 0
        
        # Pullback to 12h EMA21 (within 0.5% for entry zone)
        near_ema = abs(close[i] - ema_21_12h_aligned[i]) / ema_21_12h_aligned[i] < 0.005
        
        # Entry conditions
        long_entry = trend_up and mom_bullish and near_ema
        short_entry = trend_down and mom_bearish and near_ema
        
        # Exit conditions: trend reversal or momentum divergence
        exit_long = not trend_up or not mom_bullish
        exit_short = not trend_down or not mom_bearish
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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