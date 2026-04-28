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
    
    # Get weekly data for long-term trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA50 trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Get daily data for ATR and range
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily range (high - low)
    daily_range = high_1d - low_1d
    
    # Get 6h data for Donchian breakout
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align weekly trend to 6h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Align daily ATR and range to 6h
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    range_aligned = align_htf_to_ltf(prices, df_1d, daily_range)
    
    # Align 6h Donchian to 6h (no additional delay needed as it's already 6h data)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Calculate 6-period RSI for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_aligned[i]) or 
            np.isnan(range_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Get current weekly trend
        is_uptrend = weekly_uptrend_aligned[i] > 0.5
        is_downtrend = weekly_downtrend_aligned[i] > 0.5
        
        # Volatility filter: only trade when volatility is elevated
        # Current 6h volatility vs daily average
        current_vol = (high[i] - low[i])  # 6h range
        avg_vol = range_aligned[i] / 4.0  # approximate 6h from daily (4x 6h in day)
        vol_filter = current_vol > (avg_vol * 0.5)  # at least 50% of average 6h volatility
        
        # Breakout conditions
        bullish_breakout = close[i] > donchian_high_aligned[i]
        bearish_breakout = close[i] < donchian_low_aligned[i]
        
        # Long conditions: bullish breakout in uptrend or ranging market with volume
        long_condition = False
        if is_uptrend:
            long_condition = bullish_breakout and vol_filter and (rsi[i] < 70)
        else:
            # In downtrend or ranging, only take strong breakouts with good momentum
            long_condition = bullish_breakout and vol_filter and (rsi[i] > 50) and (rsi[i] < 60)
        
        # Short conditions: bearish breakout in downtrend or ranging market
        short_condition = False
        if is_downtrend:
            short_condition = bearish_breakout and vol_filter and (rsi[i] > 30)
        else:
            # In uptrend or ranging, only take strong breakdowns with weak momentum
            short_condition = bearish_breakout and vol_filter and (rsi[i] < 50) and (rsi[i] > 40)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite breakout or RSI extreme
        elif position == 1 and (bearish_breakout or rsi[i] > 80):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (bullish_breakout or rsi[i] < 20):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_DonchianBreakout_WeeklyTrend_VolFilter"
timeframe = "6h"
leverage = 1.0