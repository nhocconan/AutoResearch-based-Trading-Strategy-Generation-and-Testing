#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h strategy using 1d RSI extremes with 1w Supertrend trend filter
    # Works in both bull and bear: RSI <30/70 captures mean reversion,
    # 1w Supertrend defines trend direction (green=long bias, red=short bias),
    # volume confirmation ensures momentum. Discrete sizing (0.25) minimizes fee drag.
    # Target: 20-40 trades/year to stay within 4h optimal range.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for RSI (primary HTF for reversal signals)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Get 1w data for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d RSI (14-period)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    rsi = np.where(avg_loss == 0, 100, rsi)  # handle no loss case
    rsi = np.where(avg_gain == 0, 0, rsi)   # handle no gain case
    
    # Calculate 1w Supertrend (10, 3.0)
    atr_period = 10
    atr_mult = 3.0
    
    tr1 = pd.Series(high_1w).sub(pd.Series(low_1w))
    tr2 = pd.Series(high_1w).sub(pd.Series(close_1w).shift(1)).abs()
    tr3 = pd.Series(low_1w).sub(pd.Series(close_1w).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    hl2 = (pd.Series(high_1w) + pd.Series(low_1w)) / 2
    upper_band = (hl2 + atr_mult * atr).values
    lower_band = (hl2 - atr_mult * atr).values
    
    supertrend = np.zeros(len(close_1w))
    direction = np.ones(len(close_1w))  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_1w[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Get 1d volume for confirmation (20-period average)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h primary timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(supertrend_aligned[i]) or
            np.isnan(direction_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        idx_1d = i // (24 * 6)  # 1d bars in 4h timeframe (6 bars per day)
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: Supertrend direction
        uptrend = direction_aligned[i] == 1
        downtrend = direction_aligned[i] == -1
        
        # Entry conditions: RSI extremes + trend alignment + volume
        enter_long = (rsi_aligned[i] < 30) and uptrend and volume_confirmed
        enter_short = (rsi_aligned[i] > 70) and downtrend and volume_confirmed
        
        # Stoploss: based on 1d ATR
        # Calculate 1d ATR for stoploss
        tr1_1d = pd.Series(high_1d).sub(pd.Series(low_1d))
        tr2_1d = pd.Series(high_1d).sub(pd.Series(close_1d).shift(1)).abs()
        tr3_1d = pd.Series(low_1d).sub(pd.Series(close_1d).shift(1)).abs()
        tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
        atr_1d = tr_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
        
        stop_distance = 2.0 * atr_1d_aligned[i] if not np.isnan(atr_1d_aligned[i]) else np.inf
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - stop_distance
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + stop_distance
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "4h_1d_1w_rsi_extreme_supertrend_volume_v1"
timeframe = "4h"
leverage = 1.0