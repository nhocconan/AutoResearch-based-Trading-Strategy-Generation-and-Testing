#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for primary trend filter (weekly EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Get 1d data for volatility and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d volume SMA for spike detection
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Get 6h data for entry signals (price action)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 5:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # 6h price position within daily range (for mean reversion in ranging markets)
    # Calculate daily range from 1d data, but we need current day's range
    # Since we're on 6h chart, we use today's high/low from 1d data
    # For intraday calculation, we approximate using recent 1d high/low
    # Actually, we'll use 6h high/low vs 1d context for mean reversion signals
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 14)  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: weekly EMA21
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Volatility filter: avoid extremely low volatility (but allow moderate)
        vol_filter = atr_14_1d_aligned[i] > np.mean(atr_14_1d_aligned[max(0, i-30):i+1]) * 0.5
        
        # Volume filter: current 6h volume above average (not spike, just confirmation)
        vol_confirm = volume[i] > volume_sma_20_1d_aligned[i] * 0.8
        
        # Mean reversion signal: price deviation from weekly trend
        # In uptrend: look for pullbacks to weekly EMA for long entries
        # In downtrend: look for bounces to weekly EMA for short entries
        price_to_ema_ratio = close[i] / ema_21_1w_aligned[i]
        
        # Long setup: mild pullback in uptrend (price slightly below weekly EMA)
        long_setup = uptrend and (price_to_ema_ratio < 1.0) and (price_to_ema_ratio > 0.98)
        
        # Short setup: mild bounce in downtrend (price slightly above weekly EMA)
        short_setup = downtrend and (price_to_ema_ratio > 1.0) and (price_to_ema_ratio < 1.02)
        
        # Entry conditions: setup + volatility + volume
        long_entry = long_setup and vol_filter and vol_confirm
        short_entry = short_setup and vol_filter and vol_confirm
        
        # Exit conditions: return to weekly EMA or contrary signal
        if position == 1:
            # Exit long when price returns to or exceeds weekly EMA
            long_exit = close[i] >= ema_21_1w_aligned[i]
        elif position == -1:
            # Exit short when price returns to or goes below weekly EMA
            short_exit = close[i] <= ema_21_1w_aligned[i]
        else:
            long_exit = False
            short_exit = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyEMA21_Pullback_VolumeFilter"
timeframe = "6h"
leverage = 1.0