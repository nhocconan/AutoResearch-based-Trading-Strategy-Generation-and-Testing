#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper AND close > 1w EMA50 AND volume > 2.0x average
# Short when price breaks below Donchian lower AND close < 1w EMA50 AND volume > 2.0x average
# Exit when price crosses Donchian middle (mean reversion) OR trend reversal (price crosses 1w EMA50)
# Uses 6h timeframe (target: 50-150 total trades over 4 years = 12-37/year) with weekly trend filter for BTC/ETH resilience in bull/bear markets.
# Weekly EMA50 provides strong trend filter reducing whipsaw; volume spike confirms breakout authenticity.

name = "6h_Donchian20_1wEMA50_Volume_v1"
timeframe = "6h"
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
    
    # Get 6h data for Donchian calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Donchian(20) on 6h data (using previous 20 bars)
    if len(high_6h) >= 20:
        upper_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().shift(1).values
        lower_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().shift(1).values
        middle_6h = (upper_6h + lower_6h) / 2
    else:
        upper_6h = np.full_like(high_6h, np.nan)
        lower_6h = np.full_like(low_6h, np.nan)
        middle_6h = np.full_like(high_6h, np.nan)
    
    # Align Donchian levels to 6h timeframe (already aligned since calculated on 6h)
    upper_aligned = align_htf_to_ltf(prices, df_6h, upper_6h)
    lower_aligned = align_htf_to_ltf(prices, df_6h, lower_6h)
    middle_aligned = align_htf_to_ltf(prices, df_6h, middle_6h)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current 6h volume > 2.0x 20-period average (spike confirmation)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for Donchian and EMA
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > upper AND close > 1w EMA50 AND volume spike
            if close[i] > upper_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < lower AND close < 1w EMA50 AND volume spike
            elif close[i] < lower_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < middle (mean reversion) OR trend reversal (close < 1w EMA50)
            if close[i] < middle_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > middle (mean reversion) OR trend reversal (close > 1w EMA50)
            if close[i] > middle_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals