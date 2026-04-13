#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and 1d volume confirmation
    # Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe
    # Uses 1w for trend direction (price vs EMA50), 1d for volume spike, Camarilla levels from 1d for entries
    # Works in both bull and bear: trend filter avoids counter-trend trades, volume confirms momentum
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # We use H3/L3 for entries: H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    camarilla_h3 = np.zeros_like(close_1d)
    camarilla_l3 = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        camarilla_h3[i] = close_1d[i-1] + 1.125 * (high_1d[i-1] - low_1d[i-1])
        camarilla_l3[i] = close_1d[i-1] - 1.125 * (high_1d[i-1] - low_1d[i-1])
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: 1d volume > 2.0x 20-period average
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate 12h ATR for stoploss
    tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        # Get 1d bar index for current 12h bar (each 1d bar = 2 12h bars)
        idx_1d = i // 2
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        volume_confirmed = volume_1d[idx_1d] > 2.0 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: price above/below 1w EMA50
        trend_up = close[i] > ema50_1w_aligned[i]
        trend_down = close[i] < ema50_1w_aligned[i]
        
        # Entry conditions: Camarilla breakout with volume and trend confirmation
        enter_long = trend_up and volume_confirmed and close[i] > camarilla_h3_aligned[i]
        enter_short = trend_down and volume_confirmed and close[i] < camarilla_l3_aligned[i]
        
        # Stoploss: 2.5 * ATR
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.5 * atr_12h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.5 * atr_12h[i]
        
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

name = "12h_1w_1d_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0