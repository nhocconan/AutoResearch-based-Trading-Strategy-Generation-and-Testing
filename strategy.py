#!/usr/bin/env python3
"""
4h_bullish_engulfing_volume_breakout
- Long when: 12h bullish trend AND price breaks above 4h Donchian high AND bullish engulfing candle AND volume > 1.5x 20-period average
- Short when: 12h bearish trend AND price breaks below 4h Donchian low AND bearish engulfing candle AND volume > 1.5x 20-period average
- Exit on opposite Donchian break or trend reversal
- Size: 0.25
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_bullish_engulfing_volume_breakout"
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
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA trend (50-period)
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    bullish_12h = close_12h > ema_50_12h
    bearish_12h = close_12h < ema_50_12h
    
    # Align 12h trend to 4h
    bullish_12h_aligned = align_htf_to_ltf(prices, df_12h, bullish_12h.astype(float))
    bearish_12h_aligned = align_htf_to_ltf(prices, df_12h, bearish_12h.astype(float))
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    # Candlestick patterns
    body_size = np.abs(close - open_)
    upper_shadow = high - np.maximum(open_, close)
    lower_shadow = np.minimum(open_, close) - low
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulf = (
        (close > open_) &  # current candle green
        (open_ < close_) &  # previous candle red
        (open_ <= close_) &  # current open <= previous close
        (close >= open_)   # current close >= previous open
    )
    
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulf = (
        (close < open_) &  # current candle red
        (open_ > close_) &  # previous candle green
        (open_ >= close_) &  # current open >= previous close
        (close <= open_)   # current close <= previous open
    )
    
    # Shift for previous candle comparison
    open_ = prices['open'].values
    close_ = np.roll(close, 1)
    open__ = np.roll(open_, 1)
    close_[0] = np.nan
    open__[0] = np.nan
    
    bullish_engulf = (
        (close > open_) & 
        (close_ < open_) & 
        (open_ <= close_) & 
        (close >= open_)
    )
    bearish_engulf = (
        (close < open_) & 
        (close_ > open_) & 
        (open_ >= close_) & 
        (close <= open_)
    )
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Wait for Donchian to be valid
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(bullish_12h_aligned[i]) or 
            np.isnan(bearish_12h_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price breaks below Donchian low OR trend turns bearish
            if low[i] < donchian_low[i] or bearish_12h_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above Donchian high OR trend turns bullish
            if high[i] > donchian_high[i] or bullish_12h_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: bullish trend + break above Donchian high + bullish engulfing + volume spike
            if (bullish_12h_aligned[i] > 0.5 and 
                high[i] > donchian_high[i-1] and 
                bullish_engulf[i] and 
                volume[i] > vol_threshold[i]):
                position = 1
                signals[i] = 0.25
            # Short: bearish trend + break below Donchian low + bearish engulfing + volume spike
            elif (bearish_12h_aligned[i] > 0.5 and 
                  low[i] < donchian_low[i-1] and 
                  bearish_engulf[i] and 
                  volume[i] > vol_threshold[i]):
                position = -1
                signals[i] = -0.25
    
    return signals