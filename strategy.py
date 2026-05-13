#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above 12h Donchian upper band AND close > 1d EMA50 AND volume > 1.8x average
# Short when price breaks below 12h Donchian lower band AND close < 1d EMA50 AND volume > 1.8x average
# Exit when price crosses the opposite Donchian band OR trend reversal (price crosses 1d EMA50)
# Uses 12h timeframe (target: 50-150 total trades over 4 years = 12-37/year) with daily trend filter for BTC/ETH resilience.
# Donchian channels provide clear breakout levels; EMA50 filters trend; volume spike confirms breakout authenticity.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend alignment.

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
    
    # Calculate Donchian(20) on 12h data (using previous 20 bars' OHLC)
    if len(high_12h) >= 20:
        # Use rolling window on 12h data to avoid look-ahead
        high_ma_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
        low_ma_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
        # Shift by 1 to use only completed bars (previous 20 bars, not including current)
        upper_band = np.roll(high_ma_20, 1)
        lower_band = np.roll(low_ma_20, 1)
        upper_band[0] = np.nan
        lower_band[0] = np.nan
    else:
        upper_band = np.full_like(high_12h, np.nan)
        lower_band = np.full_like(low_12h, np.nan)
    
    # Align Donchian bands to 12h timeframe (already aligned since calculated on 12h)
    upper_band_aligned = align_htf_to_ltf(prices, df_12h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 12h volume > 1.8x 20-period average (spike confirmation)
    # Use 12h volume data for consistency
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after sufficient data for EMA and Donchian
        # Skip if any required data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > upper band AND close > 1d EMA50 AND volume spike
            if close[i] > upper_band_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < lower band AND close < 1d EMA50 AND volume spike
            elif close[i] < lower_band_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < lower band (mean reversion) OR trend reversal (close < 1d EMA50)
            if close[i] < lower_band_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > upper band (mean reversion) OR trend reversal (close > 1d EMA50)
            if close[i] > upper_band_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals