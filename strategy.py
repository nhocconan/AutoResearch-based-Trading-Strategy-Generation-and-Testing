# The 12h timeframe is well-suited for capturing multi-day trends while avoiding the noise and excessive trading frequency of lower timeframes.
# This strategy combines a 1-day Donchian channel breakout with volume confirmation and a 1-week ADX trend filter.
# The idea is to enter trades when price breaks out of the 1-day Donchian channel (20-period) with above-average volume,
# but only when the weekly trend is strong (ADX > 25), helping to avoid false breakouts in choppy markets.
# Exits occur when price returns to the midpoint of the Donchian channel.
# This approach aims to capture significant moves in both bull and bear markets by following established trends,
# while the volume and ADX filters help improve the quality of signals and reduce whipsaws.
# The 12h timeframe should yield a moderate number of trades (target: 50-150 over 4 years) to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Donchian channels and ATR
    daily = get_htf_data(prices, '1d')
    close_d = daily['close'].values
    high_d = daily['high'].values
    low_d = daily['low'].values
    
    # Calculate 1-day Donchian channels (20-period)
    donch_high = np.full(len(close_d), np.nan)
    donch_low = np.full(len(close_d), np.nan)
    for i in range(20, len(close_d)):
        donch_high[i] = np.max(high_d[i-20:i])
        donch_low[i] = np.min(low_d[i-20:i])
    donch_high_aligned = align_htf_to_ltf(prices, daily, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, daily, donch_low)
    
    # Calculate 1-day ATR(14) for stoploss (not used in signals but required for risk management)
    tr1 = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1]))
    tr2 = np.maximum(np.abs(low_d[1:] - close_d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Get 1-week data for ADX trend filter
    weekly = get_htf_data(prices, '1w')
    high_w = weekly['high'].values
    low_w = weekly['low'].values
    close_w = weekly['close'].values
    
    # Calculate 1-week ADX(14)
    # True Range
    tr_w1 = np.maximum(high_w[1:] - low_w[1:], np.abs(high_w[1:] - close_w[:-1]))
    tr_w2 = np.maximum(np.abs(low_w[1:] - close_w[:-1]), tr_w1)
    tr_w = np.concatenate([[np.nan], tr_w2])
    
    # Directional Movement
    up_move = high_w[1:] - high_w[:-1]
    down_move = low_w[:-1] - low_w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) / period
        # Subsequent values: smoothed = previous_smoothed - (previous_smoothed/period) + current_value
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr_w = wilders_smoothing(tr_w, 14)
    plus_di_w = 100 * wilders_smoothing(plus_dm, 14) / atr_w
    minus_di_w = 100 * wilders_smoothing(minus_dm, 14) / atr_w
    dx_w = 100 * np.abs(plus_di_w - minus_di_w) / (plus_di_w + minus_di_w)
    adx_w = wilders_smoothing(dx_w, 14)
    adx_w_aligned = align_htf_to_ltf(prices, weekly, adx_w)
    
    # Volume threshold: 1.8x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.8 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(adx_w_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Price breaks above 1d Donchian high + volume spike + strong weekly trend (ADX > 25)
        if (close[i] > donch_high_aligned[i] and 
            volume[i] > vol_threshold[i] and
            adx_w_aligned[i] > 25):
            signals[i] = 0.25
        
        # Short: Price breaks below 1d Donchian low + volume spike + strong weekly trend (ADX > 25)
        elif (close[i] < donch_low_aligned[i] and 
              volume[i] > vol_threshold[i] and
              adx_w_aligned[i] > 25):
            signals[i] = -0.25
        
        # Exit: price returns to middle of Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (donch_high_aligned[i] + donch_low_aligned[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (donch_high_aligned[i] + donch_low_aligned[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_1d1w_Donchian20_Vol1.8x_ADX25"
timeframe = "12h"
leverage = 1.0