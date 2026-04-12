#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for 34-period EMA trend filter (weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly 34-period EMA
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for Donchian(20) channels (structure)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Calculate 12-period ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr12 = np.full(n, np.nan)
    for i in range(11, n):
        atr12[i] = np.nanmean(tr[i-11:i+1])
    
    # Calculate 24-period ATR EMA for volatility regime
    atr_ema24 = np.full(n, np.nan)
    atr_series = pd.Series(atr12)
    atr_ema24_values = atr_series.ewm(span=24, adjust=False, min_periods=24).mean().values
    atr_ema24[:] = atr_ema24_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or np.isnan(atr12[i]) or np.isnan(atr_ema24[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR12 > 1.1x 24-period ATR EMA (elevated volatility)
        vol_filter = atr12[i] > atr_ema24[i] * 1.1
        
        # Trend filter: price above/below weekly 34 EMA
        price_above_ema34 = close[i] > ema34_1w_aligned[i]
        price_below_ema34 = close[i] < ema34_1w_aligned[i]
        
        # Entry conditions: Donchian breakout in direction of trend with volatility expansion
        long_breakout = close[i] > donch_high_20_aligned[i-1]  # break above previous Donchian high
        short_breakout = close[i] < donch_low_20_aligned[i-1]  # break below previous Donchian low
        
        long_entry = long_breakout and price_above_ema34 and vol_filter
        short_entry = short_breakout and price_below_ema34 and vol_filter
        
        # Exit conditions: reversal signal or volatility contraction
        long_exit = (close[i] < ema34_1w_aligned[i]) or (atr12[i] < atr_ema24[i] * 0.9)
        short_exit = (close[i] > ema34_1w_aligned[i]) or (atr12[i] < atr_ema24[i] * 0.9)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.28
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.28
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.28
            elif position == -1:
                signals[i] = -0.28
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_1d_donchian_breakout_vol_filter_v1"
timeframe = "12h"
leverage = 1.0