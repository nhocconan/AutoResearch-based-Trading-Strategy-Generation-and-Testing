#!/usr/bin/env python3
name = "12h_Donchian20_Breakout_1dTrend_1wFilter_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Donchian and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        # Trend condition: both daily and weekly EMA34 must agree
        bullish_trend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
        bearish_trend = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]
        
        if position == 0:
            # Long: Donchian breakout above upper band in bullish trend with volume
            if close[i] > donchian_high[i] and bullish_trend and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band in bearish trend with volume
            elif close[i] < donchian_low[i] and bearish_trend and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian middle or trend turns bearish
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] < donchian_mid or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian middle or trend turns bullish
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] > donchian_mid or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Donchian(20) breakouts with daily/weekly EMA34 trend alignment and volume confirmation
# - Long when price breaks above 20-day Donchian high in bullish trend (both daily and weekly EMA34 rising)
# - Short when price breaks below 20-day Donchian low in bearish trend (both daily and weekly EMA34 falling)
# - Volume confirmation (2x average) reduces false breakouts
# - Exit when price returns to Donchian middle or trend alignment breaks
# - Position size 0.25 targets ~20-50 trades/year to stay within 12h limits
# - Dual timeframe trend filter (daily + weekly) reduces whipsaws vs single timeframe
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend) markets
# - Donchian channels provide clear support/resistance with adaptive lookback period
# - Aims for 80-200 total trades over 4 years (20-50/year) to stay within limits