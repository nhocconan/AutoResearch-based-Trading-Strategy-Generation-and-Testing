#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trend filter
# Long when price breaks above Donchian upper band AND volume > 1.5x 20-bar avg AND ATR(14) > ATR(50) (trending market)
# Short when price breaks below Donchian lower band AND volume > 1.5x 20-bar avg AND ATR(14) > ATR(50)
# Exit when price retests Donchian middle band (mean reversion) or ATR(14) < ATR(50) * 0.8 (trend weakening)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 20-50 trades/year on 4h timeframe.
# Combines price channel breakouts with volume confirmation and trend strength filter to capture
# strong directional moves while avoiding false signals in ranging markets.

name = "4h_Donchian20_VolumeConfirm_ATRTrend_v3"
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
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # ATR calculation for trend filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_trend = atr_14 > atr_50  # trending when short-term ATR > long-term ATR
    atr_trend_weak = atr_14 < (atr_50 * 0.8)  # trend weakening when short ATR much < long ATR
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_atr_trend = atr_trend[i]
        curr_atr_trend_weak = atr_trend_weak[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_middle = donchian_middle[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Donchian middle OR trend weakening
            if curr_close <= curr_middle or curr_atr_trend_weak:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Donchian middle OR trend weakening
            if curr_close >= curr_middle or curr_atr_trend_weak:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND volume confirmation AND trending market
            if curr_close > curr_upper and vol_conf and curr_atr_trend:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower AND volume confirmation AND trending market
            elif curr_close < curr_lower and vol_conf and curr_atr_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals