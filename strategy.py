#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d EMA50 trend filter + ATR-based stoploss
# Long when price breaks above Donchian upper band (20-bar high) AND volume > 1.8x 20-bar avg AND close > 1d EMA50
# Short when price breaks below Donchian lower band (20-bar low) AND volume > 1.8x 20-bar avg AND close < 1d EMA50
# Exit on opposite Donchian band touch or ATR-based trailing stop (signal=0 when price < highest high since entry - 2.5*ATR for long, or price > lowest low since entry + 2.5*ATR for short)
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year on 4h timeframe.
# Donchian provides objective breakout levels, volume confirms conviction, 1d EMA50 filters counter-trend moves.
# ATR stoploss adapts to volatility and works in both bull (trailing profits) and bear (limiting drawdowns).

name = "4h_Donchian20_VolumeConfirm_1dEMA50_ATRStop_v1"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 20-bar Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(20, 14, 50)  # Donchian, ATR, and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_atr = atr[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Update highest high since entry
            if i == entry_bar:
                highest_since_entry = curr_high
            else:
                highest_since_entry = max(highest_since_entry, curr_high)
            
            # Exit conditions:
            # 1. Price touches opposite Donchian band (lower band)
            # 2. ATR-based trailing stop: price < highest_since_entry - 2.5 * ATR
            if curr_low <= curr_donchian_low or curr_close < highest_since_entry - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
                entry_bar = -1
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if i == entry_bar:
                lowest_since_entry = curr_low
            else:
                lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Exit conditions:
            # 1. Price touches opposite Donchian band (upper band)
            # 2. ATR-based trailing stop: price > lowest_since_entry + 2.5 * ATR
            if curr_high >= curr_donchian_high or curr_close > lowest_since_entry + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
                entry_bar = -1
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper band AND volume confirmation AND close > 1d EMA50
            if curr_high > curr_donchian_high and vol_conf and curr_close > curr_ema50_1d:
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = curr_high
            # Short when price breaks below Donchian lower band AND volume confirmation AND close < 1d EMA50
            elif curr_low < curr_donchian_low and vol_conf and curr_close < curr_ema50_1d:
                signals[i] = -0.25
                position = -1
                entry_bar = i
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals