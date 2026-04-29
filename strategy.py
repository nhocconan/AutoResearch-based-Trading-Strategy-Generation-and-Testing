#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation + ATR stoploss
# Long when price breaks above Donchian(20) high AND HMA(21) rising AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian(20) low AND HMA(21) falling AND volume > 1.5x 20-bar avg
# Exit when price touches Donchian(20) opposite band OR HMA reverses
# Uses discrete position sizing (0.30) to balance return and drawdown.
# Donchian captures volatility breakouts, HMA filters trend direction, volume confirms follow-through.
# Works in bull markets (upward breakouts) and bear markets (downward breakdowns).

name = "4h_Donchian20_HMA21_VolumeConfirm_Trend_v1"
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
    
    # Get 1d data for HMA(21) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate HMA(21) on 1d data
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_1d).ewm(span=half_len, adjust=False, min_periods=half_len).mean()
    wma_full = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean()
    raw_hma = 2.0 * wma_half - wma_full
    hma_21_1d = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    # Align HMA21 to 4h timeframe
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    # Calculate HMA slope (rising/falling)
    hma_slope = np.diff(hma_21_1d_aligned, prepend=hma_21_1d_aligned[0])
    
    # Calculate Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(hma_slope[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_donch_high = donch_high[i]
        curr_donch_low = donch_low[i]
        curr_hma = hma_21_1d_aligned[i]
        curr_hma_slope = hma_slope[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price touches Donchian low OR HMA slope turns negative
            if curr_close <= curr_donch_low or curr_hma_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price touches Donchian high OR HMA slope turns positive
            if curr_close >= curr_donch_high or curr_hma_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND HMA rising AND volume confirmation
            if curr_close > curr_donch_high and curr_hma_slope > 0 and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below Donchian low AND HMA falling AND volume confirmation
            elif curr_close < curr_donch_low and curr_hma_slope < 0 and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals