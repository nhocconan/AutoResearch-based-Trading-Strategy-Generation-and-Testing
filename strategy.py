#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation + ATR stoploss.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above Donchian upper(20) AND price > 1d EMA50 AND volume > 1.5 * 12h volume MA(20);
         Short when price breaks below Donchian lower(20) AND price < 1d EMA50 AND volume > 1.5 * 12h volume MA(20).
- Exit: Long exits when price crosses below Donchian lower(10) for quicker profit taking;
        Short exits when price crosses above Donchian upper(10).
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian channels provide clear structure; EMA50 filters higher-timeframe trend; volume spike confirms conviction.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with reduced whipsaws.
- Uses ATR-based stoploss to manage risk: exit if adverse move > 2.5 * ATR(14).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate ATR(14) for stoploss
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First bar has no previous close
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20 for entry, 10 for exit)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Get 12h data for volume MA(20)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track approximate entry price for stoploss
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14)  # Donchian(20), EMA50, ATR(14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or 
            np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_high_10[i]) or 
            np.isnan(donchian_low_10[i]) or 
            np.isnan(vol_ma_12h[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter from 1d EMA50
        uptrend = curr_close > ema_50_aligned[i]
        downtrend = curr_close < ema_50_aligned[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma_12h[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm:
                # Long: price breaks above Donchian upper(20)
                if curr_high > donchian_high_20[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
            elif downtrend and vol_confirm:
                # Short: price breaks below Donchian lower(20)
                if curr_low < donchian_low_20[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check exit conditions
            exit_signal = False
            
            # Stoploss: adverse move > 2.5 * ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                exit_signal = True
            # Profit take: price crosses below Donchian lower(10)
            elif curr_low < donchian_low_10[i]:
                exit_signal = True
                
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            exit_signal = False
            
            # Stoploss: adverse move > 2.5 * ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                exit_signal = True
            # Profit take: price crosses above Donchian upper(10)
            elif curr_high > donchian_high_10[i]:
                exit_signal = True
                
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Trend_VolumeConfirm_ATRStop_v1"
timeframe = "12h"
leverage = 1.0