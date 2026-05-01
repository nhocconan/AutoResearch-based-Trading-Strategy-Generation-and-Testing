#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Uses weekly Camarilla R4/S4 levels as stronger breakout confirmation to avoid false breakouts.
# Trades only in direction of 1d EMA34 trend with volume spike confirmation.
# Weekly R4/S4 breakouts indicate very strong momentum and are less prone to whipsaw.
# Works in bull (buy R4 breakout with uptrend) and bear (sell S4 breakdown with downtrend).
# Discrete position sizing 0.25 balances return and drawdown. Target: 50-150 trades over 4 years.

name = "6h_Donchian20_Breakout_WeeklyCamarillaR4S4_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly Camarilla pivot levels (R4, S4) - very strong breakout levels
    # Camarilla: based on previous week's high, low, close
    # R4 = close + (high - low) * 1.1/2
    # S4 = close - (high - low) * 1.1/2
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    camarilla_r4 = prev_weekly_close + (prev_weekly_high - prev_weekly_low) * 1.1 / 2
    camarilla_s4 = prev_weekly_close - (prev_weekly_high - prev_weekly_low) * 1.1 / 2
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # 6h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume on 6h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20) + 1  # 35 (for EMA34 and Donchian20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Donchian breakout conditions
        breakout_up = curr_high > donchian_high[i]  # Break above upper Donchian
        breakdown_down = curr_low < donchian_low[i]  # Break below lower Donchian
        
        # Weekly Camarilla R4/S4 confirmation
        breakout_r4 = curr_close > camarilla_r4_aligned[i]  # Confirm above weekly R4
        breakdown_s4 = curr_close < camarilla_s4_aligned[i]  # Confirm below weekly S4
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND weekly R4 confirmation AND uptrend AND volume confirmation
            if breakout_up and breakout_r4 and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown down AND weekly S4 confirmation AND downtrend AND volume confirmation
            elif breakdown_down and breakdown_s4 and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown (reversal signal)
            if curr_low < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout (reversal signal)
            if curr_high > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals