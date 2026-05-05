#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band AND 1d close > 1d EMA34 AND volume > 2.0x 20-period average
# Short when price breaks below 4h Donchian lower band AND 1d close < 1d EMA34 AND volume > 2.0x 20-period average
# Exit when price crosses 1d EMA34 (trend reversal) OR Donchian middle band (mean reversion)
# Uses tighter volume filter (2.0x vs 1.8x) to reduce trades and avoid overtrading
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian channels adapt to volatility, EMA34 provides smooth trend filter, volume spike confirms breakout validity

name = "4h_Donchian20_Breakout_1dEMA34_Trend_Volume_Tight"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Donchian channels (20-period)
    if len(high) >= 20:
        # Donchian upper band: 20-period high
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Donchian lower band: 20-period low
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Donchian middle band: average of upper and lower
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average (tighter than 1.8x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND 1d close > 1d EMA34 AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND 1d close < 1d EMA34 AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA34 (trend reversal) OR below Donchian middle band (mean reversion)
            if close[i] < ema_34_1d_aligned[i] or close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA34 (trend reversal) OR above Donchian middle band (mean reversion)
            if close[i] > ema_34_1d_aligned[i] or close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals