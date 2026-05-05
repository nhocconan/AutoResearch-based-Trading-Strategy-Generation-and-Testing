#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation
# Long when: price breaks above Donchian upper channel (20-period) AND 1d EMA34 up AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian lower channel (20-period) AND 1d EMA34 down AND volume > 1.5x 20-period MA
# Exit when: price returns to Donchian midpoint (mean reversion) OR volume drops below average
# Uses Donchian for structure, 1d EMA for higher-timeframe trend, volume for conviction
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_DonchianBreakout_1dEMA34_VolumeConfirm"
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
    
    # Calculate Donchian Channel on 4h (20-period)
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Get 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # EMA trend direction: 1 = up, -1 = down, 0 = unclear
    ema_trend = np.zeros(len(ema_34_1d), dtype=int)
    for i in range(1, len(ema_34_1d)):
        if not np.isnan(ema_34_1d[i]) and not np.isnan(ema_34_1d[i-1]):
            if ema_34_1d[i] > ema_34_1d[i-1]:
                ema_trend[i] = 1
            elif ema_34_1d[i] < ema_34_1d[i-1]:
                ema_trend[i] = -1
            else:
                ema_trend[i] = ema_trend[i-1]  # hold previous trend
    
    # Align 1d EMA trend to 4h timeframe
    ema_trend_aligned = align_htf_to_ltf(prices, df_1d, ema_trend.astype(float))
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(ema_trend_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + EMA up + volume filter
            if (close[i] > donchian_high[i] and 
                ema_trend_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + EMA down + volume filter
            elif (close[i] < donchian_low[i] and 
                  ema_trend_aligned[i] == -1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR volume drops below average
            if (close[i] >= donchian_mid[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR volume drops below average
            if (close[i] <= donchian_mid[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals