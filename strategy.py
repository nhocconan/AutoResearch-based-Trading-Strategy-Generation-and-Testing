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
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate Donchian(20) channels on 1d
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf  # First bar has no previous close
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d RSI(14) for momentum filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(sma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Price relative to Donchian channels
        above_donchian_high = close[i] > donchian_high_aligned[i]
        below_donchian_low = close[i] < donchian_low_aligned[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_1d_aligned[i] > np.nanpercentile(atr_1d_aligned[:i+1], 30)
        
        # Momentum filter: RSI not extreme
        rsi_not_overbought = rsi_14_aligned[i] < 70
        rsi_not_oversold = rsi_14_aligned[i] > 30
        
        # Weekly trend filter
        above_weekly_sma = close[i] > sma_20_1w_aligned[i]
        below_weekly_sma = close[i] < sma_20_1w_aligned[i]
        
        # Entry conditions
        long_entry = above_donchian_high and vol_filter and rsi_not_overbought and above_weekly_sma
        short_entry = below_donchian_low and vol_filter and rsi_not_oversold and below_weekly_sma
        
        # Exit conditions: opposite signal or volatility expansion
        exit_long = position == 1 and (below_donchian_low or atr_1d_aligned[i] > 2.0 * atr_1d_aligned[i-1])
        exit_short = position == -1 and (above_donchian_high or atr_1d_aligned[i] > 2.0 * atr_1d_aligned[i-1])
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_vol_rsi_weekly_filter"
timeframe = "4h"
leverage = 1.0