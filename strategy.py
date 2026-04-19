#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Engulfing_Pullback_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend context (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for long-term trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily ATR for stop loss (14-period)
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        prev_close = close[i-1]
        prev_open = prices['open'].iloc[i-1]
        prev_high = prices['high'].iloc[i-1]
        prev_low = prices['low'].iloc[i-1]
        ema200 = ema200_1w_aligned[i]
        atr = atr14[i]
        
        if position == 0:
            # Bullish engulfing: green candle fully engulfs prior red candle
            bullish_engulf = (close[i] > prev_open) and (prev_close > prev_open) and \
                           (close[i] >= prev_high) and (prev_close <= prev_low)
            # Pullback to EMA200: price touches or crosses EMA200 during pullback
            pullback_to_ema = low[i] <= ema200 <= high[i]
            
            if bullish_engulf and pullback_to_ema and price > ema200:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit: close below EMA200 or ATR trailing stop
            if close[i] < ema200 or close[i] < highest_since_entry - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Track highest close since entry for trailing stop
                if i == start_idx or position == 0:
                    highest_since_entry = close[i]
                else:
                    highest_since_entry = max(highest_since_entry, close[i])
    
    # Initialize tracking variables
    if 'highest_since_entry' not in locals():
        highest_since_entry = 0
    
    return signals

# Fix: move tracking inside loop properly
def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend context (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for long-term trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily ATR for stop loss (14-period)
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        prev_close = close[i-1]
        prev_open = prices['open'].iloc[i-1]
        prev_high = prices['high'].iloc[i-1]
        prev_low = prices['low'].iloc[i-1]
        ema200 = ema200_1w_aligned[i]
        atr = atr14[i]
        
        if position == 0:
            # Bullish engulfing: green candle fully engulfs prior red candle
            bullish_engulf = (close[i] > prev_open) and (prev_close < prev_open) and \
                           (close[i] >= prev_high) and (prev_close <= prev_low)
            # Pullback to EMA200: price touches or crosses EMA200 during pullback
            pullback_to_ema = low[i] <= ema200 <= high[i]
            
            if bullish_engulf and pullback_to_ema and price > ema200:
                signals[i] = 0.25
                position = 1
                highest_since_entry = close[i]
        
        elif position == 1:
            # Exit: close below EMA200 or ATR trailing stop
            if close[i] < ema200 or close[i] < highest_since_entry - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0
            else:
                signals[i] = 0.25
                highest_since_entry = max(highest_since_entry, close[i])
    
    return signals

# Final version with corrected engulfing condition
def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend context (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for long-term trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily ATR for stop loss (14-period)
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    highest_since_entry = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        prev_close = close[i-1]
        prev_open = prices['open'].iloc[i-1]
        prev_high = prices['high'].iloc[i-1]
        prev_low = prices['low'].iloc[i-1]
        ema200 = ema200_1w_aligned[i]
        atr = atr14[i]
        
        if position == 0:
            # Bullish engulfing: current green candle fully engulfs prior red candle
            bullish_engulf = (close[i] > prev_open) and (prev_close < prev_open) and \
                           (close[i] >= prev_high) and (prev_close <= prev_low)
            # Pullback to EMA200: price touches or crosses EMA200 during pullback
            pullback_to_ema = low[i] <= ema200 <= high[i]
            
            if bullish_engulf and pullback_to_ema and price > ema200:
                signals[i] = 0.25
                position = 1
                highest_since_entry = close[i]
        
        elif position == 1:
            # Exit: close below EMA200 or ATR trailing stop
            if close[i] < ema200 or close[i] < highest_since_entry - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0
            else:
                signals[i] = 0.25
                highest_since_entry = max(highest_since_entry, close[i])
    
    return signals

# Final clean version
def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend context (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for long-term trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily ATR for stop loss (14-period)
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    highest_since_entry = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        prev_close = close[i-1]
        prev_open = prices['open'].iloc[i-1]
        prev_high = prices['high'].iloc[i-1]
        prev_low = prices['low'].iloc[i-1]
        ema200 = ema200_1w_aligned[i]
        atr = atr14[i]
        
        if position == 0:
            # Bullish engulfing: current green candle fully engulfs prior red candle
            bullish_engulf = (close[i] > prev_open) and (prev_close < prev_open) and \
                           (close[i] >= prev_high) and (prev_close <= prev_low)
            # Pullback to EMA200: price touches or crosses EMA200 during pullback
            pullback_to_ema = low[i] <= ema200 <= high[i]
            
            if bullish_engulf and pullback_to_ema and close[i] > ema200:
                signals[i] = 0.25
                position = 1
                highest_since_entry = close[i]
        
        elif position == 1:
            # Exit: close below EMA200 or ATR trailing stop
            if close[i] < ema200 or close[i] < highest_since_entry - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0
            else:
                signals[i] = 0.25
                highest_since_entry = max(highest_since_entry, close[i])
    
    return signals

# Corrected engulfing condition (prior candle must be red)
def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend context (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for long-term trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily ATR for stop loss (14-period)
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    highest_since_entry = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        prev_close = close[i-1]
        prev_open = prices['open'].iloc[i-1]
        prev_high = prices['high'].iloc[i-1]
        prev_low = prices['low'].iloc[i-1]
        ema200 = ema200_1w_aligned[i]
        atr = atr14[i]
        
        if position == 0:
            # Bullish engulfing: current candle is green and fully engulfs prior red candle
            curr_green = close[i] > prices['open'].iloc[i]
            prev_red = prev_close < prev_open
            engulfs = (close[i] >= prev_high) and (prices['open'].iloc[i] <= prev_low)
            
            if curr_green and prev_red and engulfs and low[i] <= ema200 <= high[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry = close[i]
        
        elif position == 1:
            # Exit: close below EMA200 or ATR trailing stop
            if close[i] < ema200 or close[i] < highest_since_entry - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0
            else:
                signals[i] = 0.25
                highest_since_entry = max(highest_since_entry, close[i])
    
    return signals

# Final version with correct engulfing definition
def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_prices = prices['open'].values
    
    # Get weekly data for trend context (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for long-term trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily ATR for stop loss (14-period)
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    highest_since_entry = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        prev_close = close[i-1]
        prev_open = open_prices[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        ema200 = ema200_1w_aligned[i]
        atr = atr14[i]
        
        if position == 0:
            # Bullish engulfing: current candle is green and fully engulfs prior red candle
            curr_green = close[i] > open_prices[i]
            prev_red = prev_close < prev_open
            engulfs = (close[i] >= prev_high) and (open_prices[i] <= prev_low)
            pullback_to_ema = low[i] <= ema200 <= high[i]
            
            if curr_green and prev_red and engulfs and pullback_to_ema:
                signals[i] = 0.25
                position = 1
                highest_since_entry = close[i]
        
        elif position == 1:
            # Exit: close below EMA200 or ATR trailing stop
            if close[i] < ema200 or close[i] < highest_since_entry - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0
            else:
                signals[i] = 0.25
                highest_since_entry = max(highest_since_entry, close[i])
    
    return signals

# Add volume confirmation to reduce false signals
def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_prices = prices['open'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend context (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for long-term trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily ATR for stop loss (14-period)
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    highest_since_entry = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(atr14[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        prev_close = close[i-1]
        prev_open = open_prices[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        ema200 = ema200_1w_aligned[i]
        atr = atr14[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Bullish engulfing: current candle is green and fully engulfs prior red candle
            curr_green = close[i] > open_prices[i]
            prev_red = prev_close < prev_open
            engulfs = (close[i] >= prev_high) and (open_prices[i] <= prev_low)
            pullback_to_ema = low[i] <= ema200 <= high[i]
            volume_confirmed = vol > 1.5 * vol_ma
            
            if curr_green and prev_red and engulfs and pullback_to_ema and volume_confirmed:
                signals[i] = 0.25
                position = 1
                highest_since_entry = close[i]
        
        elif position == 1:
            # Exit: close below EMA200 or ATR trailing stop
            if close[i] < ema200 or close[i] < highest_since_entry - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0
            else:
                signals[i] = 0.25
                highest_since_entry = max(highest_since_entry, close[i])
    
    return signals

# Final optimization: increase volume threshold and add cooldown
def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_prices = prices['open'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend context (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for long-term trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily ATR for stop loss (14-period)
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    highest_since_entry = 0
    bars_since_exit = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(atr14[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        prev_close = close[i-1]
        prev_open = open_prices[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        ema200 = ema200_1w_aligned[i]
        atr = atr14[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0 and bars_since_exit >= 5:  # 5-bar cooldown
            # Bullish engulfing: current candle is green and fully engulfs prior red candle
            curr_green = close[i] > open_prices[i]
            prev_red = prev_close < prev_open
            engulfs = (close[i] >= prev_high) and (open_prices[i] <= prev_low)
            pullback_to_ema = low[i] <= ema200 <= high[i]
            volume_confirmed = vol > 2.0 * vol_ma
            
            if curr_green and prev_red and engulfs and pullback_to_ema and volume_confirmed:
                signals[i] = 0.25
                position = 1
                highest_since_entry = close[i]
                bars_since_exit = 0
        
        elif position == 1:
            # Exit: close below EMA200 or ATR trailing stop
            if close[i] < ema200 or close[i] < highest_since_entry - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
                highest_since_entry = max(highest_since_entry, close[i])
                bars_since_exit += 1
        else:
            bars_since_exit += 1
    
    return signals

# Final version with correct cooldown logic
def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_prices = prices['open'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend context (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for long-term trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily ATR for stop loss (14-period)
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    highest_since_entry = 0
    bars_since_exit = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(atr14[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            if position == 0:
                bars_since_exit += 1
            else:
                bars_since_exit = 0
            continue
        
        prev_close = close[i-1]
        prev_open = open_prices[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        ema200 = ema200_1w_aligned[i]
        atr = atr14[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            if bars_since_exit >= 5:  # 5-bar cooldown
                # Bullish engulfing: current candle is green and fully engulfs prior red candle
                curr_green = close[i] > open_prices[i]
                prev_red = prev_close < prev_open
                engulfs = (close[i] >= prev_high) and (open_prices[i] <= prev_low)
                pullback_to_ema = low[i] <= ema200 <= high[i]
                volume_confirmed = vol > 2.0 * vol_ma
                
                if curr_green and prev_red and engulfs and pullback_to_ema and volume_confirmed:
                    signals[i] = 0.25
                    position = 1
                    highest_since_entry = close[i]
                    bars_since_exit = 0
            else:
                bars_since_exit += 1
        
        elif position == 1:
            # Exit: close below EMA200 or ATR trailing stop
            if close[i] < ema200 or close[i] < highest_since_entry - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
                highest_since_entry = max(highest_since_entry, close[i])
                bars_since_exit = 0
    
    return signals

# Ensure we reset bars_since_exit on entry
def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_prices = prices['open'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend context (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for long-term trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily ATR for stop loss (14-period)
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    highest_since_entry = 0
    bars_since_exit = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(atr14[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            if position == 0:
                bars_since_exit += 1
            else:
                bars_since_exit = 0
            continue
        
        prev_close = close[i-1]
        prev_open = open_prices[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        ema200 = ema200_1w_aligned[i]
        atr = atr14[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            if bars_since_exit >= 5:  # 5-bar cooldown
                # Bullish engulfing: current candle is green and fully engulfs prior red candle
                curr_green = close[i] > open_prices[i]
                prev_red = prev_close < prev_open
                engulfs = (close[i] >= prev_high) and (open_prices[i] <= prev_low)
                pullback_to_ema = low[i] <= ema200 <= high[i]
                volume_confirmed = vol