#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and ADX(14) > 25 trend filter
# - Long when price breaks above Donchian upper channel(20) + 1d volume > 2.0x 20-period volume SMA + ADX(14) > 25
# - Short when price breaks below Donchian lower channel(20) + 1d volume > 2.0x 20-period volume SMA + ADX(14) > 25
# - Exit: price returns to Donchian middle (mean of upper/lower) or ATR-based stoploss
# - Position sizing: 0.25 discrete level
# - Donchian breakouts capture volatility expansion, volume confirms institutional participation, ADX filters choppy markets
# - Works in bull/bear: breakouts effective in both regimes when volatility compresses then expands
# - 12h timeframe targets 12-37 trades/year with strict entry conditions to minimize fee drag

name = "12h_1d_donchian_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 12h Donchian Channels (20)
    dc_period = 20
    upper_dc = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    lower_dc = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    middle_dc = (upper_dc + lower_dc) / 2.0
    
    # Calculate 12h ADX(14) for trend filter
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    # Plus Directional Movement
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    plus_dm[0] = 0
    # Minus Directional Movement
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    minus_dm[0] = 0
    # Smoothed values
    atr = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    # Handle division by zero and invalid values
    plus_di = np.where(atr == 0, 0, plus_di)
    minus_di = np.where(atr == 0, 0, minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = np.where(np.isnan(adx) | np.isinf(adx), 0, adx)
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # ATR for dynamic stoploss (2x ATR)
    atr_multiplier = 2.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or np.isnan(adx[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period SMA (strong spike)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
        vol_confirm = vol_1d_current[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market (good for breakouts)
        trending_market = adx[i] > 25
        
        # Donchian Channel conditions
        dc_upper = upper_dc[i]
        dc_lower = lower_dc[i]
        dc_middle = middle_dc[i]
        price = close[i]
        
        # Entry conditions: price breaks Donchian bands with volume and trend confirmation
        long_entry = (price > dc_upper) and vol_confirm and trending_market
        short_entry = (price < dc_lower) and vol_confirm and trending_market
        
        # Exit conditions: price returns to middle Donchian channel
        long_exit = price < dc_middle
        short_exit = price > dc_middle
        
        # ATR-based stoploss
        if position == 1:  # Long position
            # Calculate entry price approximation (we don't track exact entry, use recent close)
            # Simplified: use 5-bar lookback for entry estimation
            lookback = min(5, i)
            entry_approx = np.mean(close[i-lookback:i]) if lookback > 0 else close[i]
            stop_loss = entry_approx - (atr_multiplier * atr[i])
            stop_exit = price < stop_loss
        elif position == -1:  # Short position
            lookback = min(5, i)
            entry_approx = np.mean(close[i-lookback:i]) if lookback > 0 else close[i]
            stop_loss = entry_approx + (atr_multiplier * atr[i])
            stop_exit = price > stop_loss
        else:
            stop_exit = False
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit or stop_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit or stop_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals