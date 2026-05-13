#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_Trend_Volume
Hypothesis: Weekly Donchian breakout with daily trend and volume confirmation works in both bull and bear markets.
Breakout above 4-week high with daily uptrend and volume spike = long.
Breakdown below 4-week low with daily downtrend and volume spike = short.
Exit on opposite band touch or trend reversal. Uses weekly trend filter for higher timeframe bias.
Target: 10-25 trades/year per symbol.
"""

name = "1d_WeeklyDonchian_Breakout_Trend_Volume"
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
    
    # Daily trend: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_daily = close > ema_50
    downtrend_daily = close < ema_50
    
    # Weekly Donchian (20 periods = ~4 weeks)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate weekly Donchian channels (20-period high/low)
    donchian_high = np.zeros(len(high_weekly))
    donchian_low = np.zeros(len(low_weekly))
    for i in range(20, len(high_weekly)):
        donchian_high[i] = np.max(high_weekly[i-20:i])
        donchian_low[i] = np.min(low_weekly[i-20:i])
    
    # Align to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Weekly trend: EMA50 on weekly
    ema_50_weekly = pd.Series(df_weekly['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_weekly = df_weekly['close'].values > ema_50_weekly
    downtrend_weekly = df_weekly['close'].values < ema_50_weekly
    uptrend_weekly_aligned = align_htf_to_ltf(prices, df_weekly, uptrend_weekly)
    downtrend_weekly_aligned = align_htf_to_ltf(prices, df_weekly, downtrend_weekly)
    
    # Volume confirmation: volume > 2.0 * 20-day average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        daily_uptrend = uptrend_daily[i]
        daily_downtrend = downtrend_daily[i]
        weekly_uptrend = uptrend_weekly_aligned[i]
        weekly_downtrend = downtrend_weekly_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above weekly Donchian high, daily uptrend, weekly uptrend, volume confirmation
            if close[i] > upper_band and daily_uptrend and weekly_uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below weekly Donchian low, daily downtrend, weekly downtrend, volume confirmation
            elif close[i] < lower_band and daily_downtrend and weekly_downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch weekly Donchian low or daily trend turns down
            if close[i] < lower_band or not daily_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch weekly Donchian high or daily trend turns up
            if close[i] > upper_band or not daily_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals