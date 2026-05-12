#/usr/bin/env python3
# 4h_Donchian20_TrendFilter_Breakout_1dEMA34_Volume
# Hypothesis: On 4h timeframe, enter long when price breaks above Donchian upper band (20-period high) with trend filter (price > daily EMA34) and volume confirmation (volume > 1.5x 20-period average). Enter short when price breaks below Donchian lower band (20-period low) with price < daily EMA34 and volume confirmation. Exit when price crosses back through daily EMA34 (trend reversal). Uses daily EMA34 for trend filter to work in both bull (breakouts with trend) and bear (counter-trend reversals at EMA). Targets ~25 trades/year for low fee drag.

name = "4h_Donchian20_TrendFilter_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 55  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        upper_band = high_roll[i]
        lower_band = low_roll[i]
        ema1d_trend = ema34_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band with trend filter and volume confirmation
            if close[i] > upper_band and close[i] > ema1d_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower band with trend filter and volume confirmation
            elif close[i] < lower_band and close[i] < ema1d_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below daily EMA34 (trend reversal)
            if close[i] < ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above daily EMA34 (trend reversal)
            if close[i] > ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals