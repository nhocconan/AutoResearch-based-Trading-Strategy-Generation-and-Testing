#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trend filter
# Long when price breaks above Donchian upper(20) AND volume > 1.5x 20-bar avg AND ATR(14) > ATR(50) (trending market)
# Short when price breaks below Donchian lower(20) AND volume > 1.5x 20-bar avg AND ATR(14) > ATR(50)
# Exit when price retests Donchian midpoint or opposite breakout level
# Uses discrete position sizing (0.30) to balance return and drawdown. Target: 20-50 trades/year on 4h timeframe.
# Donchian channels provide clear structure, volume confirms breakout strength, ATR ratio ensures trending conditions.
# Works in bull via breakout continuation, in bear via breakdown continuation. Novelty: ATR regime filter reduces whipsaw.

name = "4h_Donchian20_VolumeConfirm_ATRTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # ATR-based trend filter: ATR(14) > ATR(50) indicates trending market
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # First TR
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_trend = atr_14 > atr_50  # True when shorter ATR > longer ATR (trending)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for ATR(50) and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        trend_conf = atr_trend[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_mid = donchian_mid[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests midpoint or breaks below lower band
            if curr_low <= curr_mid or curr_close <= curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price retests midpoint or breaks above upper band
            if curr_high >= curr_mid or curr_close >= curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper band AND volume confirmation AND trending market
            if curr_high > curr_upper and vol_conf and trend_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below lower band AND volume confirmation AND trending market
            elif curr_low < curr_lower and vol_conf and trend_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals