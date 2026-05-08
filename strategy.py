#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 1-Week Momentum Reversal
# Uses 1-week RSI to detect overbought/oversold conditions on longer timeframe.
# Enters mean-reversion trades on 6h when price reaches Bollinger Bands (20,2) 
# in the direction opposite to weekly RSI extreme.
# Weekly RSI > 70 = short bias, wait for 6h price to touch upper BB.
# Weekly RSI < 30 = long bias, wait for 6h price to touch lower BB.
# Volume confirmation: current 6h volume > 1.5x 20-period 6h average.
# Works in both bull/bear markets as it fades extremes rather than following trends.
# Target: 60-120 total trades over 4 years (15-30/year).

name = "6h_WeeklyRSI_BollingerMeanReversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(weekly_close, np.nan)
    avg_loss = np.full_like(weekly_close, np.nan)
    
    # Wilder's smoothing
    for i in range(len(weekly_close)):
        if i == 0:
            avg_gain[i] = np.mean(gain[1:15]) if len(gain) >= 15 else np.nan
            avg_loss[i] = np.mean(loss[1:15]) if len(loss) >= 15 else np.nan
        else:
            if not np.isnan(avg_gain[i-1]) and not np.isnan(avg_loss[i-1]):
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
            else:
                avg_gain[i] = np.nan
                avg_loss[i] = np.nan
    
    weekly_rsi = np.full_like(weekly_close, np.nan)
    for i in range(len(weekly_close)):
        if not np.isnan(avg_loss[i]) and avg_loss[i] != 0:
            weekly_rsi[i] = 100 - (100 / (1 + avg_gain[i] / avg_loss[i]))
        elif not np.isnan(avg_gain[i]) and avg_loss[i] == 0:
            weekly_rsi[i] = 100
        else:
            weekly_rsi[i] = 0
    
    # Get 6h data for Bollinger Bands and volume
    if len(close) < 20:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2) on 6h
    ma_20 = np.full(n, np.nan)
    std_20 = np.full(n, np.nan)
    upper_bb = np.full(n, np.nan)
    lower_bb = np.full(n, np.nan)
    
    for i in range(20, n):
        ma_20[i] = np.mean(close[i-20:i])
        std_20[i] = np.std(close[i-20:i])
        upper_bb[i] = ma_20[i] + 2 * std_20[i]
        lower_bb[i] = ma_20[i] - 2 * std_20[i]
    
    # Volume average (20-period) on 6h
    vol_avg_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align weekly RSI to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_weekly, weekly_rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 6h volume > 1.5x 20-period average
        vol_filter = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: price at Bollinger Band in opposite direction of weekly RSI extreme
            # Long when weekly RSI < 30 (oversold) and price touches lower BB
            long_condition = (
                rsi_aligned[i] < 30 and
                close[i] <= lower_bb[i] and
                vol_filter
            )
            
            # Short when weekly RSI > 70 (overbought) and price touches upper BB
            short_condition = (
                rsi_aligned[i] > 70 and
                close[i] >= upper_bb[i] and
                vol_filter
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle (20-day MA) or RSI normalizes
            if close[i] >= ma_20[i] or rsi_aligned[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle (20-day MA) or RSI normalizes
            if close[i] <= ma_20[i] or rsi_aligned[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals