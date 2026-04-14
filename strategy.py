# 4h Candlestick Pattern + Volume + Trend Strategy
# This strategy combines bullish/bearish engulfing patterns with volume confirmation and ADX trend filter
# Engulfing patterns signal potential reversals, volume confirms conviction, ADX ensures trend context
# Works in both bull and bear markets by trading with the trend on pullbacks
# Target: 20-40 trades/year per symbol to minimize fee drag

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
    
    # Load 1d data ONCE for ADX (trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX (14 periods)
    adx_len = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Candlestick patterns: bullish and bearish engulfing
    bullish_engulf = (close > open_) & (open_ < close_) & \
                     (close > open_.shift(1)) & (open_ < close_.shift(1)) & \
                     ((close - open_) > (open_.shift(1) - close_.shift(1)))
    bearish_engulf = (close < open_) & (open_ > close_) & \
                     (open_ > close_.shift(1)) & (close < open_.shift(1)) & \
                     ((open_ - close_) > (close_.shift(1) - open_.shift(1)))
    
    # Since we don't have open in values, calculate from prices
    open_ = prices['open'].values
    
    bullish_engulf = (close > open_) & (open_ < close_) & \
                     (close > np.roll(open_, 1)) & (open_ < np.roll(close_, 1)) & \
                     ((close - open_) > (np.roll(open_, 1) - np.roll(close_, 1)))
    bearish_engulf = (close < open_) & (open_ > close_) & \
                     (open_ > np.roll(close_, 1)) & (close < np.roll(open_, 1)) & \
                     ((open_ - close_) > (np.roll(close_, 1) - np.roll(open_, 1)))
    
    # Handle first element
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, adx_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 20 indicates trending market (lower threshold for more signals)
        trending = adx_aligned[i] > 20
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: bullish engulfing + volume + trend
            if (bullish_engulf[i] and 
                volume_confirmed and 
                trending):
                position = 1
                signals[i] = position_size
            # Enter short: bearish engulfing + volume + trend
            elif (bearish_engulf[i] and 
                  volume_confirmed and 
                  trending):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish engulfing or loss of momentum
            if bearish_engulf[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bullish engulfing or loss of momentum
            if bullish_engulf[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Candlestick_Pattern_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0