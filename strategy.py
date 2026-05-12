#!/usr/bin/env python3
# 1d Weekly Donchian Breakout with Volume Confirmation and Volatility Filter
# Hypothesis: Weekly Donchian channels capture long-term trend structure.
# Breakouts above weekly high or below weekly low with volume confirmation
# and low volatility (ATR ratio) filter capture strong moves while avoiding
# whipsaws in choppy markets. Works in bull markets via breakouts and
# bear markets via breakdowns. Designed for very low trade frequency
# (~10-25/year) to minimize fee drag on 1d timeframe.

name = "1d_WeeklyDonchian_Volume_VolatilityFilter"
timeframe = "1d"
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
    
    # === Weekly Data for Donchian Channels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:  # Need at least 21 periods for Donchian(20) + 1
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_volume = df_1w['volume'].values
    
    # Weekly Donchian Channels (20-period)
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (available after weekly bar closes)
    donchian_high_daily = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_daily = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Weekly Volume Average (20-period) for confirmation
    vol_ma_weekly = pd.Series(weekly_volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_weekly_daily = align_htf_to_ltf(prices, df_1w, vol_ma_weekly)
    
    # === Daily Volatility Filter (ATR Ratio) ===
    # True Range
    tr1 = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    
    # ATR(10) and ATR(30)
    atr10 = pd.Series(tr1).rolling(window=10, min_periods=10).mean().values
    atr30 = pd.Series(tr1).rolling(window=30, min_periods=30).mean().values
    
    # Avoid division by zero
    atr_ratio = np.where(atr30 > 0, atr10 / atr30, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_daily[i]) or np.isnan(donchian_low_daily[i]) or
            np.isnan(vol_ma_weekly_daily[i]) or np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above weekly Donchian high + volume surge + low volatility (trending market)
            if (close[i] > donchian_high_daily[i] and
                volume[i] > vol_ma_weekly_daily[i] * 1.5 and
                atr_ratio[i] < 0.4):  # Low ATR ratio indicates trending market
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly Donchian low + volume surge + low volatility
            elif (close[i] < donchian_low_daily[i] and
                  volume[i] > vol_ma_weekly_daily[i] * 1.5 and
                  atr_ratio[i] < 0.4):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to weekly Donchian low (mean reversion within channel)
            if close[i] <= donchian_low_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly Donchian high
            if close[i] >= donchian_high_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals