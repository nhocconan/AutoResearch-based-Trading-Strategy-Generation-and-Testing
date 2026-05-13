#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above upper Donchian(20) AND close > 1d EMA50 AND volume > 1.8x average
# Short when price breaks below lower Donchian(20) AND close < 1d EMA50 AND volume > 1.8x average
# Exit when price crosses the Donchian middle (mean reversion) OR trend reversal (price crosses 1d EMA50)
# Uses 12h timeframe (target: 50-150 total trades over 4 years = 12-37/year) with daily trend filter.
# Donchian channels provide structural breakouts; EMA50 filters trend; volume spike confirms authenticity.

name = "12h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 12h data for Donchian calculation (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian(20) on 12h data (using previous bar's OHLC to avoid look-ahead)
    if len(high_12h) >= 20:
        # Rolling window on previous bars only
        upper_donch = pd.Series(high_12h).rolling(window=20, min_periods=20).max().shift(1).values
        lower_donch = pd.Series(low_12h).rolling(window=20, min_periods=20).min().shift(1).values
        middle_donch = (upper_donch + lower_donch) / 2.0
    else:
        upper_donch = np.full_like(high_12h, np.nan)
        lower_donch = np.full_like(low_12h, np.nan)
        middle_donch = np.full_like(high_12h, np.nan)
    
    # Align Donchian levels to 12h timeframe (already aligned since calculated on 12h)
    upper_donch_aligned = align_htf_to_ltf(prices, df_12h, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_12h, lower_donch)
    middle_donch_aligned = align_htf_to_ltf(prices, df_12h, middle_donch)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 12h volume > 1.8x 20-period average (spike confirmation)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after sufficient data for EMA and volume
        # Skip if any required data is NaN
        if (np.isnan(upper_donch_aligned[i]) or np.isnan(lower_donch_aligned[i]) or np.isnan(middle_donch_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > upper Donchian AND close > 1d EMA50 AND volume spike
            if close[i] > upper_donch_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < lower Donchian AND close < 1d EMA50 AND volume spike
            elif close[i] < lower_donch_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < middle Donchian (mean reversion) OR trend reversal (close < 1d EMA50)
            if close[i] < middle_donch_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > middle Donchian (mean reversion) OR trend reversal (close > 1d EMA50)
            if close[i] > middle_donch_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals