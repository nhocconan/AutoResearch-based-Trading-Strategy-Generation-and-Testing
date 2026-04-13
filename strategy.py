#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla H4/L4 breakout with 1d EMA50 trend filter
    # Works in both bull/bear: Camarilla captures 12h reversals, 1d EMA50 filters false breakouts
    # Target: 12-37 trades/year to minimize fee drag (50-150 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_volume = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate Camarilla levels for each day (H4 and L4 levels)
    camarilla_h4 = daily_close + (daily_high - daily_low) * 1.1 / 2
    camarilla_l4 = daily_close - (daily_high - daily_low) * 1.1 / 2
    
    # Get 1d EMA50 for trend filter
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d volume 20-period average for confirmation
    vol_avg_20_1d = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        idx_1d = i // 2  # 12h bars per day = 2
        if idx_1d >= len(daily_volume):
            signals[i] = 0.0
            continue
        volume_confirmed = daily_volume[idx_1d] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Trend direction from 1d EMA(50)
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: Camarilla level break + trend + volume
        enter_long = (close[i] > camarilla_h4_aligned[i]) and trend_up and volume_confirmed
        enter_short = (close[i] < camarilla_l4_aligned[i]) and trend_down and volume_confirmed
        
        # ATR-based stoploss (using 12h ATR)
        if i >= 14:
            tr = np.maximum(np.maximum(high[i] - low[i], np.abs(high[i] - close[i-1])), np.abs(low[i] - close[i-1]))
            atr_12h = np.sqrt(np.mean(np.maximum(np.maximum(
                high[i-13:i+1] - low[i-13:i+1],
                np.abs(high[i-13:i+1] - np.roll(high[i-13:i+1], 1))[1:]),
                np.abs(low[i-13:i+1] - np.roll(high[i-13:i+1], 1))[1:]
            )))
            atr_12h = max(atr_12h, 1e-8)  # avoid division by zero
        else:
            atr_12h = np.nan
        
        if np.isnan(atr_12h):
            atr_12h = 0.01 * close[i]  # fallback
        
        # Stoploss conditions: 2.5 * ATR
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.5 * atr_12h
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.5 * atr_12h
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]  # record entry price at close (filled next bar open)
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]  # record entry price at close (filled next bar open)
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

name = "12h_1d_camarilla_pivot_breakout_trend_volume_v1"
timeframe = "12h"
leverage = 1.0