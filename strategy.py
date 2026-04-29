#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND price > 12h EMA50 AND volume > 1.8x 20-bar avg
# Short when price breaks below Donchian(20) low AND price < 12h EMA50 AND volume > 1.8x 20-bar avg
# Exit when price retests Donchian(20) midpoint (mean reversion in range) or opposite band breakout
# Uses discrete position sizing (0.25) to reduce fee drag and improve test generalization.
# Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years) to avoid overtrading.
# Donchian channels provide objective volatility-based breakout levels that work in both trending and ranging markets.
# 12h EMA50 provides HTF trend filter to avoid counter-trend trades in bear markets.
# Volume confirmation ensures breakouts have conviction, reducing false signals.
# Works in bull markets by capturing upward breakouts and in bear markets by shorting downward breakdowns.

name = "4h_Donchian20_12hEMA50_VolumeConfirm_v1"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: >1.8x 20-bar average volume (balanced to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_dch_high = donchian_high[i]
        curr_dch_low = donchian_low[i]
        curr_dch_mid = donchian_mid[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Donchian midpoint OR breaks below lower band (stop and reverse)
            if curr_close <= curr_dch_mid or curr_close < curr_dch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Donchian midpoint OR breaks above upper band (stop and reverse)
            if curr_close >= curr_dch_mid or curr_close > curr_dch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND price > 12h EMA50 AND volume confirmation
            if curr_close > curr_dch_high and curr_close > curr_ema50_12h and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND price < 12h EMA50 AND volume confirmation
            elif curr_close < curr_dch_low and curr_close < curr_ema50_12h and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals