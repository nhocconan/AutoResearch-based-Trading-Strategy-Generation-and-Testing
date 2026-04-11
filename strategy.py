#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1w ADX trend filter
# - Long: price breaks above Donchian(20) high, volume > 1.5x 20-period avg, 1w ADX(14) > 25
# - Short: price breaks below Donchian(20) low, volume > 1.5x 20-period avg, 1w ADX(14) > 25
# - Exit: ATR trailing stop (2.0 * ATR) or opposite Donchian breakout
# - Uses discrete position sizing: ±0.30 to balance return and drawdown
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Donchian breakouts capture momentum, volume confirmation reduces false signals,
#   1w ADX filter ensures we only trade in strong weekly trends, reducing whipsaw in ranging markets

name = "4h_1d_1w_donchian_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Load 1w data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr_1w = np.maximum(high_1w - low_1w, np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    tr_14 = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1w ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute ATR for trailing stop (4h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(atr_14[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Donchian(20) channels (based on previous 20 periods)
        lookback_start = max(0, i - 20)
        lookback_end = i  # exclude current bar
        if lookback_end - lookback_start < 20:
            # Not enough history for Donchian channels
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
            
        donchian_high = np.max(high[lookback_start:lookback_end])
        donchian_low = np.min(low[lookback_start:lookback_end])
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_lookback_start = max(0, i - 20)
        vol_lookback_end = i
        if vol_lookback_end - vol_lookback_start < 20:
            vol_confirm = False
        else:
            vol_avg = np.mean(volume[vol_lookback_start:vol_lookback_end])
            vol_confirm = volume_current > 1.5 * vol_avg
        
        # Trend filter: 1w ADX > 25 (indicates trending market)
        adx_trend = adx_aligned[i] > 25
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Donchian high, volume confirmation, trending
        if close_price > donchian_high and vol_confirm and adx_trend:
            enter_long = True
        
        # Short breakout: price below Donchian low, volume confirmation, trending
        if close_price < donchian_low and vol_confirm and adx_trend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, high_price)
            # Exit long if ATR trailing stop hit or price breaks below Donchian low
            exit_long = (close_price <= highest_since_entry - 2.0 * atr_14[i]) or (close_price < donchian_low)
        elif position == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, low_price)
            # Exit short if ATR trailing stop hit or price breaks above Donchian high
            exit_short = (close_price >= lowest_since_entry + 2.0 * atr_14[i]) or (close_price > donchian_high)
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
            highest_since_entry = high_price
            lowest_since_entry = low_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.30
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals