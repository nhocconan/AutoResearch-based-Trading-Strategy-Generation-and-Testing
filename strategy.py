#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot levels from daily data with volume confirmation and choppiness regime filter
# Long when price touches or crosses above Camarilla L3 level, price > 12h EMA200 (uptrend), volume > 1.5x 12h avg volume, and choppiness > 61.8 (ranging market)
# Short when price touches or crosses below Camarilla H3 level, price < 12h EMA200 (downtrend), volume > 1.5x 12h avg volume, and choppiness > 61.8 (ranging market)
# Exit when price reaches Camarilla H4/L4 levels or trend reverses (price crosses EMA200 opposite direction)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses daily Camarilla levels for mean reversion in ranging markets, filtered by trend and volume
# Designed for choppy/range-bound markets (2025 conditions) with infrequent, high-probability trades

name = "12h_camarilla_1d_chop_vol_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    camarilla_H4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_H3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_L3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_L4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # 12h data for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_200 = pd.Series(close_12h).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200)
    
    # 12h volume average for confirmation
    volume_12h = df_12h['volume'].values
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    # Choppiness Index (14) for regime detection
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        atr_sum = np.zeros(len(close_arr))
        true_range = np.zeros(len(close_arr))
        
        for i in range(len(close_arr)):
            if i == 0:
                true_range[i] = high_arr[i] - low_arr[i]
            else:
                true_range[i] = max(
                    high_arr[i] - low_arr[i],
                    abs(high_arr[i] - close_arr[i-1]),
                    abs(low_arr[i] - close_arr[i-1])
                )
            
            if i < window:
                atr_sum[i] = np.nan
            else:
                atr_sum[i] = np.sum(true_range[i-window+1:i+1])
        
        # Calculate Chop
        chop = np.full(len(close_arr), np.nan)
        for i in range(window-1, len(close_arr)):
            if atr_sum[i] > 0 and np.sum(true_range[i-window+1:i+1]) > 0:
                max_close = np.max(close_arr[i-window+1:i+1])
                min_close = np.min(close_arr[i-window+1:i+1])
                if max_close != min_close:
                    log_val = np.log10(atr_sum[i] / (max_close - min_close))
                    chop[i] = 100 * log_val / np.log10(window)
        
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_ma_12h_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches H4 or trend reverses (price below EMA200)
            elif close[i] >= H4_aligned[i] or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches L4 or trend reverses (price above EMA200)
            elif close[i] <= L4_aligned[i] or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation, trend alignment, and chop filter
            # Long: price touches/crosses above L3, price > EMA200 (uptrend), volume spike, chop > 61.8 (ranging)
            if (close[i] >= L3_aligned[i] and
                close[i] > ema_200_aligned[i] and
                volume[i] > 1.5 * volume_ma_12h_aligned[i] and
                chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price touches/crosses below H3, price < EMA200 (downtrend), volume spike, chop > 61.8 (ranging)
            elif (close[i] <= H3_aligned[i] and
                  close[i] < ema_200_aligned[i] and
                  volume[i] > 1.5 * volume_ma_12h_aligned[i] and
                  chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals