#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_Trend_Volume_v2
Hypothesis: Weekly Donchian channel breakouts with daily trend filter and volume confirmation capture major trends while avoiding whipsaws in both bull and bear markets.
Long when price breaks above weekly Donchian(20) high with daily uptrend and volume spike.
Short when price breaks below weekly Donchian(20) low with daily downtrend and volume spike.
Exit on opposite Donchian band touch or trend reversal. Weekly trend adds higher timeframe bias.
Target: 15-25 trades/year per symbol.
"""

name = "1d_WeeklyDonchian_Breakout_Trend_Volume_v2"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly Donchian Channel: 20-period high/low
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    donchian_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Daily trend: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_daily = close > ema_50
    downtrend_daily = close < ema_50
    
    # Weekly trend filter: EMA50
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_weekly = df_1w['close'].values > ema_50_1w
    downtrend_weekly = df_1w['close'].values < ema_50_1w
    uptrend_weekly_aligned = align_htf_to_ltf(prices, df_1w, uptrend_weekly)
    downtrend_weekly_aligned = align_htf_to_ltf(prices, df_1w, downtrend_weekly)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        don_high = donchian_high_aligned[i]
        don_low = donchian_low_aligned[i]
        uptrend_d = uptrend_daily[i]
        downtrend_d = downtrend_daily[i]
        uptrend_w = uptrend_weekly_aligned[i]
        downtrend_w = downtrend_weekly_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above weekly Donchian high, daily uptrend, weekly uptrend filter, volume confirmation
            if close[i] > don_high and uptrend_d and uptrend_w and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below weekly Donchian low, daily downtrend, weekly downtrend filter, volume confirmation
            elif close[i] < don_low and downtrend_d and downtrend_w and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch weekly Donchian low or daily trend turns down
            if close[i] < don_low or not uptrend_d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch weekly Donchian high or daily trend turns up
            if close[i] > don_high or not downtrend_d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals