#!/usr/bin/env python3
"""
1d_Donchian_Breakout_WeeklyEMA34_Trend_VolumeConfirm
Hypothesis: Daily Donchian(20) breakout with weekly EMA34 trend filter and volume confirmation.
Designed for 30-100 trades over 4 years (7-25/year) to minimize fee drag on 1d timeframe.
Uses weekly EMA34 for trend alignment to work in both bull and bear markets.
ATR-based stoploss for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA34 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA34 trend filter (loaded ONCE)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily data for Donchian calculation (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Prior daily bar OHLC for Donchian calculation (20-period)
    # We need to calculate Donchian on daily data, then align to 1d timeframe
    # Since our primary timeframe is 1d, we can calculate directly
    # But to be safe with alignment, we'll calculate on df_1d and use directly
    
    # Calculate Donchian channels on daily data (20-period)
    # Using the daily dataframe directly since we're on 1d timeframe
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # ATR for stoploss calculation
    tr0 = np.abs(high - low)
    tr1 = np.abs(high[1:] - close[:-1])
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr0[0]], np.maximum(tr1, tr2)])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for weekly EMA (34), Donchian (20), volume MA (20), ATR (14)
    start_idx = max(34, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume spike + weekly EMA34 trend alignment
            long_breakout = curr_high > donchian_high[i]
            short_breakout = curr_low < donchian_low[i]
            
            # Trend filter: price must be on correct side of weekly EMA34
            long_trend = curr_close > ema_34_1w_aligned[i]
            short_trend = curr_close < ema_34_1w_aligned[i]
            
            long_entry = (long_breakout and volume_spike[i] and long_trend)
            short_entry = (short_breakout and volume_spike[i] and short_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below Donchian low (failed breakout) 
            # or trend reverses or ATR stoploss hit
            atr_stop = entry_price - 2.5 * atr[i]
            if curr_close < donchian_low[i] or curr_close < ema_34_1w_aligned[i] or curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Donchian high (failed breakout) 
            # or trend reverses or ATR stoploss hit
            atr_stop = entry_price + 2.5 * atr[i]
            if curr_close > donchian_high[i] or curr_close > ema_34_1w_aligned[i] or curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_WeeklyEMA34_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0