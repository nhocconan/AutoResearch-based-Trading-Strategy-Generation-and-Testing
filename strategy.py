#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_Trend_Confirmation
Hypothesis: Buy when price breaks above weekly Donchian high (20-week) with price above 200-day EMA and volume above average. Sell when price breaks below weekly Donchian low or closes below 200-day EMA. Designed to capture major trends while avoiding whipsaws in ranging markets. Weekly trend filter ensures alignment with long-term direction, reducing false breakouts. Target 10-20 trades per year on daily timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly Donchian channels (20-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    highest_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high = align_htf_to_ltf(prices, df_1w, highest_high)
    donchian_low = align_htf_to_ltf(prices, df_1w, lowest_low)
    
    # === Daily 200-period EMA for trend filter ===
    close = prices['close'].values
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # === Volume confirmation: 20-day average volume ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(ema_200[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_open = prices['open'].values[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high + above 200 EMA + volume confirmation
            if (price_close > donchian_high[i] and
                price_close > ema_200[i] and
                vol_ma_20[i] > 0 and
                vol_current > vol_ma_20[i]):
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below weekly Donchian low OR closes below 200 EMA
            elif (price_close < donchian_low[i] or
                  price_close < ema_200[i]):
                signals[i] = -0.30
                position = -1
        
        elif position != 0:
            # Exit long: Price breaks below weekly Donchian low OR closes below 200 EMA
            if position == 1:
                if (price_close < donchian_low[i] or
                    price_close < ema_200[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            # Exit short: Price breaks above weekly Donchian high OR closes above 200 EMA
            elif position == -1:
                if (price_close > donchian_high[i] or
                    price_close > ema_200[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Trend_Confirmation"
timeframe = "1d"
leverage = 1.0