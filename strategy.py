#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h strategy using weekly Camarilla pivot levels for structure and 1d trend filter
    # Weekly Camarilla provides robust support/resistance that works in both bull and bear markets
    # 1d EMA filter avoids trading against the daily trend, reducing false breakouts
    # Volume confirmation ensures momentum behind moves
    # Target: 12-30 trades/year to minimize fee drag on 6h timeframe
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1w data for weekly Camarilla pivot calculation (primary HTF for structure)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly OHLC
    weekly_open = df_1w['open'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Get daily OHLC
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_volume = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate weekly Camarilla levels (H3 and L3 - more conservative than H4/L4)
    camarilla_h3_weekly = weekly_close + (weekly_high - weekly_low) * 1.1 / 4
    camarilla_l3_weekly = weekly_close - (weekly_high - weekly_low) * 1.1 / 4
    
    # Get 1d data for trend filter (EMA 50)
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d volume for confirmation (20-period average)
    vol_avg_20_1d = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    camarilla_h3_weekly_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_weekly)
    camarilla_l3_weekly_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_weekly)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    atr_multiplier = 2.0  # ATR stoploss multiplier
    
    # Calculate 6h ATR for stoploss (using true range)
    tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_6h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_weekly_aligned[i]) or 
            np.isnan(camarilla_l3_weekly_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.2x 20-period average
        idx_1d = i // 4  # 6h bars in 1d timeframe (4 bars/day)
        if idx_1d >= len(daily_volume):
            signals[i] = 0.0
            continue
        volume_confirmed = daily_volume[idx_1d] > 1.2 * vol_avg_20_1d_aligned[i]
        
        # Trend direction from 1d EMA(50)
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: Weekly Camarilla level break + trend + volume
        enter_long = (close[i] > camarilla_h3_weekly_aligned[i]) and trend_up and volume_confirmed
        enter_short = (close[i] < camarilla_l3_weekly_aligned[i]) and trend_down and volume_confirmed
        
        # Stoploss conditions
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - atr_multiplier * atr_6h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + atr_multiplier * atr_6h[i]
        
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

name = "6h_1w_1d_camarilla_pivot_breakout_trend_volume_v1"
timeframe = "6h"
leverage = 1.0