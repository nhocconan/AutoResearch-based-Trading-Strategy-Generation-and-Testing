# 12h_WeeklyCamarilla_R4S4_Breakout_Volume - Weekly pivot breakout with volume confirmation and monthly trend filter
# Hypothesis: Weekly Camarilla R4/S4 levels act as strong support/resistance. Breaking these with volume confirmation
# and aligned with monthly trend captures major moves while avoiding false breakouts. Works in bull/bear by
# following monthly trend direction. Targets 15-25 trades/year to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get monthly data for trend filter
    df_1M = get_htf_data(prices, '1M')
    if len(df_1M) < 30:
        return np.zeros(n)
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Monthly trend: EMA50
    close_1M_series = pd.Series(df_1M['close'].values)
    ema50_1M = close_1M_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1M_aligned = align_htf_to_ltf(prices, df_1M, ema50_1M)
    
    # Weekly range for pivot calculations (previous week)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_range = weekly_high - weekly_low
    
    # Weekly Camarilla pivot levels (R4/S4)
    camarilla_r4 = weekly_close + weekly_range * 1.1 / 2
    camarilla_s4 = weekly_close - weekly_range * 1.1 / 2
    
    # Align weekly pivot levels to 12h
    r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Volume filter: above 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema50_1M_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to monthly EMA50
        trend_up = close[i] > ema50_1M_aligned[i]
        trend_down = close[i] < ema50_1M_aligned[i]
        
        # Volume filter
        vol_filter = volume[i] > vol_ma[i]
        
        # Entry conditions: break of weekly R4/S4 with volume and trend alignment
        long_entry = (close[i] > r4_aligned[i]) and vol_filter and trend_up
        short_entry = (close[i] < s4_aligned[i]) and vol_filter and trend_down
        
        # Exit conditions: return to opposite pivot level
        long_exit = (close[i] < s4_aligned[i])
        short_exit = (close[i] > r4_aligned[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WeeklyCamarilla_R4S4_Breakout_Volume"
timeframe = "12h"
leverage = 1.0