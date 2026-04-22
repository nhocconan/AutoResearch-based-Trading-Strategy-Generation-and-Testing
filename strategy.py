#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian channel breakout with weekly EMA50 trend filter and volume confirmation
    # Donchian channels provide clear breakout signals with built-in trend following
    # Weekly EMA50 filters for primary trend direction to avoid counter-trend trades
    # Volume confirmation ensures breakouts have institutional participation
    # This combination works in both bull (breakouts continue) and bear (breakdowns continue) markets
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for EMA50 trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian channel (20-period) on daily timeframe
    # Upper band = 20-period high, Lower band = 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x average volume
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper + weekly uptrend + volume spike
            if close[i] > donchian_upper[i] and close[i] > ema50_1w_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + weekly downtrend + volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema50_1w_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle of Donchian channel or trend reversal
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if position == 1:
                if close[i] < donchian_middle or close[i] < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_middle or close[i] > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian_20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0