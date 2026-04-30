#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Point (PP) breakout with 1d EMA50 trend filter and volume confirmation
# Weekly PP provides key support/resistance from prior week - breakouts indicate momentum shifts
# 1d EMA50 provides medium-term trend filter to avoid counter-trend trades
# Volume confirmation (>1.4x average) ensures breakout legitimacy
# Works in bull/bear: breakouts occur in all regimes, volume confirms legitimacy, trend filter reduces false signals
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_WeeklyPP_Breakout_1dEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Weekly Pivot Point (PP), R1, S1 from previous week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly PP = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # Weekly R1 = (2 * PP) - Prior Week Low
    # Weekly S1 = (2 * PP) - Prior Week High
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = (2 * pp) - weekly_low
    s1 = (2 * pp) - weekly_high
    
    # Align to 6h timeframe (previous week's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.4x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.4 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_pp = pp_aligned[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above Weekly R1 + above 1d EMA50
                if curr_close > curr_r1 and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below Weekly S1 + below 1d EMA50
                elif curr_close < curr_s1 and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below Weekly PP (reversal) or above Weekly R1 + 0.5*(R1-S1) (take profit)
            take_profit_level = curr_r1 + 0.5 * (curr_r1 - curr_s1)
            if curr_close < curr_pp or curr_close > take_profit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Weekly PP (reversal) or below Weekly S1 - 0.5*(R1-S1) (take profit)
            take_profit_level = curr_s1 - 0.5 * (curr_r1 - curr_s1)
            if curr_close > curr_pp or curr_close < take_profit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals