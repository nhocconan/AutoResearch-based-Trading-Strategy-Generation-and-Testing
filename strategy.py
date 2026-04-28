#!/usr/bin/env python3
"""
1d_WeeklyKeltner_Breakout_TrendFilter_Volume
Hypothesis: On 1d, buy when price breaks above Keltner upper band in uptrend (price > weekly EMA200) with volume confirmation; sell when breaks below lower band in downtrend. Weekly trend filter ensures trading with major trend. Targets 15-25 trades/year to minimize fee drag and avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and Keltner bands
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_w = df_w['close'].values
    ema_200_w = pd.Series(close_w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_w_aligned = align_htf_to_ltf(prices, df_w, ema_200_w)
    
    # Calculate weekly Keltner channels (20, 1.5)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # EMA20 of typical price
    tp_w = (high_w + low_w + close_w) / 3
    ema_tp_w = pd.Series(tp_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR of weekly
    tr1_w = high_w - low_w
    tr2_w = np.abs(high_w - np.roll(close_w, 1))
    tr3_w = np.abs(low_w - np.roll(close_w, 1))
    tr2_w[0] = np.inf
    tr3_w[0] = np.inf
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    atr_w = pd.Series(tr_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner bands
    upper_w = ema_tp_w + (1.5 * atr_w)
    lower_w = ema_tp_w - (1.5 * atr_w)
    
    # Align to daily
    ema_200_w_aligned = align_htf_to_ltf(prices, df_w, ema_200_w)
    upper_w_aligned = align_htf_to_ltf(prices, df_w, upper_w)
    lower_w_aligned = align_htf_to_ltf(prices, df_w, lower_w)
    
    # Volume confirmation: 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_w_aligned[i]) or np.isnan(upper_w_aligned[i]) or 
            np.isnan(lower_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from weekly EMA200
        trend_up = close[i] > ema_200_w_aligned[i]
        trend_down = close[i] < ema_200_w_aligned[i]
        
        # Price relative to weekly Keltner bands
        price_above_upper = close[i] > upper_w_aligned[i]
        price_below_lower = close[i] < lower_w_aligned[i]
        
        # Entry logic:
        # Long: Break above upper band in uptrend with volume
        long_entry = trend_up and price_above_upper and vol_confirm[i]
        # Short: Break below lower band in downtrend with volume
        short_entry = trend_down and price_below_lower and vol_confirm[i]
        
        # Exit logic: Opposite band or trend reversal
        long_exit = (close[i] < lower_w_aligned[i]) or (not trend_up)
        short_exit = (close[i] > upper_w_aligned[i]) or (not trend_down)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyKeltner_Breakout_TrendFilter_Volume"
timeframe = "1d"
leverage = 1.0