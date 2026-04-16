#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when price breaks above 20-day high with 1w EMA50 rising and volume > 1.5x average
# Short when price breaks below 20-day low with 1w EMA50 falling and volume > 1.5x average
# Uses 1d timeframe with 1w trend filter to avoid counter-trend trades
# Target: 20-80 total trades over 4 years (5-20/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (primary timeframe) ===
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1w data (higher timeframe for trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_prev = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().shift(1).values
    ema50_1w_prev = np.where(np.isnan(ema50_1w_prev), ema50_1w, ema50_1w_prev)
    ema50_rising = ema50_1w > ema50_1w_prev
    ema50_falling = ema50_1w < ema50_1w_prev
    
    # 1w volume average for confirmation
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 1d timeframe
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema50_falling)
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i]) or
            np.isnan(vol_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            atr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
            atr[0] = high[0] - low[0]  # Fix first value
            atr_ma = pd.Series(atr).rolling(window=14, min_periods=14).mean().values
            if i < len(atr_ma) and not np.isnan(atr_ma[i]):
                if price < entry_price - 2.5 * atr_ma[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
        
        elif position == -1:  # Short position
            atr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
            atr[0] = high[0] - low[0]  # Fix first value
            atr_ma = pd.Series(atr).rolling(window=14, min_periods=14).mean().values
            if i < len(atr_ma) and not np.isnan(atr_ma[i]):
                if price > entry_price + 2.5 * atr_ma[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below 10-day low or trend weakens
            lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
            if i < len(lowest_low_10) and not np.isnan(lowest_low_10[i]):
                if price < lowest_low_10[i] or not ema50_rising_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above 10-day high or trend weakens
            highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
            if i < len(highest_high_10) and not np.isnan(highest_high_10[i]):
                if price > highest_high_10[i] or not ema50_falling_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume confirmation and trend alignment
            if vol > 1.5 * vol_ma_20_1w_aligned[i]:
                # Breakout above 20-day high with rising 1w EMA50
                if price > highest_high[i] and ema50_rising_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Breakdown below 20-day low with falling 1w EMA50
                elif price < lowest_low[i] and ema50_falling_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1wEMA50_TrendVolumeFilter_v1"
timeframe = "1d"
leverage = 1.0