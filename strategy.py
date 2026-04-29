#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and ATR-based volume confirmation
# Long when price breaks above 20-bar high AND price > 1w EMA50 AND volume > 1.5x ATR-scaled volume
# Short when price breaks below 20-bar low AND price < 1w EMA50 AND volume > 1.5x ATR-scaled volume
# Exit on opposite Donchian(10) break (more responsive exit) or ATR trailing stop (3x ATR)
# Uses discrete position sizing (0.25) to control drawdown and fee drag.
# Target: 12-37 trades/year on 6h (50-150 total over 4 years).
# 1w EMA50 provides strong trend filter effective in both bull and bear markets.
# ATR-scaled volume confirmation adapts to volatility regimes, reducing false breakouts.
# Donchian channels work well in crypto trends with clear breakout/breakdown behavior.

name = "6h_Donchian20_1wEMA50_ATRVolume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for volatility-based volume scaling and stop loss
    # Using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate ATR-scaled volume average (20-period)
    atr_volume = volume * atr  # Volume weighted by volatility
    atr_volume_series = pd.Series(atr_volume)
    atr_volume_ma_20 = atr_volume_series.rolling(window=20, min_periods=20).mean().values
    # Avoid division by zero
    atr_volume_ma_20_safe = np.where(atr_volume_ma_20 == 0, np.nan, atr_volume_ma_20)
    volume_confirm = atr_volume > 1.5 * atr_volume_ma_20_safe
    
    # Calculate Donchian channels
    # Donchian(20) for entry
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian(10) for exit (more responsive)
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    # For ATR trailing stop
    highest_high_since_entry = np.full(n, np.nan)
    lowest_low_since_entry = np.full(n, np.nan)
    
    start_idx = max(50, 20, 14)  # EMA50, Donchian20, ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(high_10[i]) or np.isnan(low_10[i]) or np.isnan(atr[i]) or
            np.isnan(atr_volume_ma_20_safe[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_50 = ema_50_1w_aligned[i]
        atr_val = atr[i]
        
        # Donchian levels
        upper_20 = high_20[i]
        lower_20 = low_20[i]
        upper_10 = high_10[i]
        lower_10 = low_10[i]
        
        # Update trailing stop levels
        if position == 1:  # Long position
            if i == start_idx or position == 0:  # New entry or just flipped
                highest_high_since_entry[i] = curr_high
            else:
                highest_high_since_entry[i] = max(highest_high_since_entry[i-1], curr_high)
        elif position == -1:  # Short position
            if i == start_idx or position == 0:  # New entry or just flipped
                lowest_low_since_entry[i] = curr_low
            else:
                lowest_low_since_entry[i] = min(lowest_low_since_entry[i-1], curr_low)
        else:  # Flat
            highest_high_since_entry[i] = np.nan
            lowest_low_since_entry[i] = np.nan
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit 1: price breaks below Donchian(10) low
            # Exit 2: ATR trailing stop (3x ATR below highest high since entry)
            exit_condition = (curr_close < lower_10) or \
                           (not np.isnan(highest_high_since_entry[i]) and 
                            curr_close < highest_high_since_entry[i] - 3.0 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit 1: price breaks above Donchian(10) high
            # Exit 2: ATR trailing stop (3x ATR above lowest low since entry)
            exit_condition = (curr_close > upper_10) or \
                           (not np.isnan(lowest_low_since_entry[i]) and 
                            curr_close > lowest_low_since_entry[i] + 3.0 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume confirmation
            if curr_close > upper_20 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry[i] = curr_high
            # Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume confirmation
            elif curr_close < lower_20 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry[i] = curr_low
            else:
                signals[i] = 0.0
    
    return signals