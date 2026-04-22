#!/usr/bin/env python3

"""
Hypothesis: Weekly Donchian Channel Breakout with Daily Volume Confirmation and ATR Stop
Trade long when price breaks above weekly Donchian upper band with daily volume confirmation,
short when breaks below lower band. Uses ATR-based volatility filter to avoid breakouts in
extreme volatility. Weekly trend provides direction, daily volume confirms momentum.
Designed for low trade frequency (7-25 trades/year) with ATR stop to manage risk in
both bull and bear markets. Weekly timeframe reduces noise, daily volume ensures
institutional participation.
"""

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
    
    # Daily ATR(14) for volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_series = pd.Series(true_range)
    atr = tr_series.rolling(window=14, min_periods=14).mean().values
    
    # Load weekly data for Donchian channels - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly Donchian Channel (20 periods)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_dc_upper = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_dc_lower = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily
    dc_upper_aligned = align_htf_to_ltf(prices, df_weekly, weekly_dc_upper)
    dc_lower_aligned = align_htf_to_ltf(prices, df_weekly, weekly_dc_lower)
    
    # Daily volume confirmation: volume > 1.3x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter: price above/below weekly 50 EMA
    weekly_close = df_weekly['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(dc_upper_aligned[i]) or np.isnan(dc_lower_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(weekly_ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid extreme volatility (ATR > 3x 50-day average)
        if i >= 50:
            atr_ma_50 = pd.Series(atr[:i+1]).rolling(window=50, min_periods=1).mean().iloc[-1]
            vol_filter = atr[i] < 3 * atr_ma_50
        else:
            vol_filter = True
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper + weekly uptrend + vol confirm + vol filter
            if (close[i] > dc_upper_aligned[i] and 
                weekly_ema50_aligned[i] > weekly_ema50_aligned[i-1] and
                vol_confirm and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower + weekly downtrend + vol confirm + vol filter
            elif (close[i] < dc_lower_aligned[i] and 
                  weekly_ema50_aligned[i] < weekly_ema50_aligned[i-1] and
                  vol_confirm and vol_filter):
                signals[i] = -0.25
                position = -1
        else:
            # ATR-based trailing stop and exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: ATR stop or price returns to weekly Donchian middle
                atr_stop = close[i] <= dc_upper_aligned[i] - 2.0 * atr[i]
                mid_line = (dc_upper_aligned[i] + dc_lower_aligned[i]) / 2
                mean_revert = close[i] < mid_line
                if atr_stop or mean_revert:
                    exit_signal = True
            else:  # position == -1
                # Exit short: ATR stop or price returns to weekly Donchian middle
                atr_stop = close[i] >= dc_lower_aligned[i] + 2.0 * atr[i]
                mid_line = (dc_upper_aligned[i] + dc_lower_aligned[i]) / 2
                mean_revert = close[i] > mid_line
                if atr_stop or mean_revert:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Weekly_Donchian_Breakout_DailyVol_ATRStop"
timeframe = "1d"
leverage = 1.0