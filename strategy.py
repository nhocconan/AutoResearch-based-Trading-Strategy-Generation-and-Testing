#!/usr/bin/env python3
# Hypothesis: 4h Donchian channel breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper (20) AND close > 12h EMA50 AND volume > 2.0x average
# Short when price breaks below Donchian lower (20) AND close < 12h EMA50 AND volume > 2.0x average
# Exit when price crosses Donchian middle (mean reversion) OR trend reversal (price crosses 12h EMA50)
# Uses 4h timeframe (target: 75-200 total trades over 4 years = 19-50/year) with 12h trend filter for BTC/ETH resilience.
# Donchian provides clear structure; 12h EMA50 filters trend; volume spike confirms breakout authenticity.

name = "4h_Donchian20_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
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
    
    # Get 4h data for Donchian calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels (20-period) on 4h data (using previous bar's OHLC to avoid look-ahead)
    if len(high_4h) >= 20:
        # Use rolling window on previous bar's data
        high_series = pd.Series(high_4h)
        low_series = pd.Series(low_4h)
        donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
        donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full_like(high_4h, np.nan)
        donchian_low = np.full_like(low_4h, np.nan)
        donchian_mid = np.full_like(high_4h, np.nan)
    
    # Align Donchian levels to 4h timeframe (already aligned since calculated on 4h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Get 12h data for EMA50 trend filter (HTF as specified)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current 4h volume > 2.0x 20-period average (spike confirmation)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for EMA and volume
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > Donchian high AND close > 12h EMA50 AND volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < Donchian low AND close < 12h EMA50 AND volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < Donchian mid (mean reversion) OR trend reversal (close < 12h EMA50)
            if close[i] < donchian_mid_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > Donchian mid (mean reversion) OR trend reversal (close > 12h EMA50)
            if close[i] > donchian_mid_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals