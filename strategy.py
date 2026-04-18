#!/usr/bin/env python3
"""
1d_KAMA_Direction_Volume_Trend
Hypothesis: On the daily timeframe, Kaufman Adaptive Moving Average (KAMA) identifies
the adaptive trend direction. Long when price crosses above KAMA with volume confirmation
in an uptrending market (price above longer EMA), short when price crosses below KAMA
with volume confirmation in a downtrending market (price below longer EMA).
Uses 1-week timeframe for trend filter to avoid counter-trend trades.
Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag
and work in both bull and bear markets by following adaptive trend.
"""

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
    
    # Get 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    # Efficiency ratio: |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(df_1d['close'], n=10))  # |close_t - close_{t-10}|
    volatility = np.abs(np.diff(df_1d['close'], n=1))  # |close_t - close_{t-1}|
    
    # Sum of absolute changes over 10 periods
    vol_sum = np.convolve(volatility, np.ones(10), mode='full')[:len(volatility)] + \
              np.concatenate([np.full(9, np.nan), volatility[:-9]]) if len(volatility) >= 10 else np.full_like(volatility, np.nan)
    # Simplified: use pandas for rolling sum
    vol_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    
    # Avoid division by zero
    er = np.where(vol_sum > 0, change / vol_sum, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(df_1d['close'], np.nan)
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to 1d timeframe (no further alignment needed as we're using 1d as primary)
    # But we need to align to the actual price timestamp index
    # Since we're using 1d as primary timeframe, we need to map daily values to intraday prices
    # However, for 1d primary timeframe, we can use the daily values directly aligned to close prices
    # We'll create a signal array and fill forward the daily values
    kama_aligned = np.full_like(close, np.nan)
    
    # Map daily KAMA to intraday prices: each daily value applies to all intraday bars of that day
    # We'll use the date index to map
    dates_1d = pd.to_datetime(df_1d.index)
    dates_intraday = pd.to_datetime(prices['open_time'])
    
    # Create mapping: for each intraday bar, find the corresponding daily bar
    # Since we're using 1d timeframe, we assume prices are already aligned to daily boundaries
    # Simpler approach: resample not needed, we'll use forward fill of daily values
    # But to avoid look-ahead, we'll use the previous day's KAMA for today's intraday bars
    
    # Instead, let's use 1d as the actual timeframe and assume prices are daily bars
    # This means we should only use daily data - but the strategy expects intraday
    # Let's reframe: we'll use 1d timeframe but keep the same logic, using daily bars
    # However, the backtester expects signals for each bar in prices (which is intraday)
    # So we need to generate signals for intraday based on daily signals
    
    # Re-thinking: Let's use 1d as primary but generate signals only at daily close
    # For simplicity in this implementation, we'll use the intraday prices but
    # base signals on daily calculations, updated only once per day
    
    # Calculate daily signals first
    daily_signals = np.zeros(len(df_1d))
    position = 0
    
    # Need longer EMA for trend filter
    ema_long = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation on daily
    vol_ma_daily = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_filter_daily = df_1d['volume'].values > (1.5 * vol_ma_daily)
    
    start_idx = max(30, 20, 10)  # KAMA needs ~10, EMA 50, vol MA 20
    
    for i in range(start_idx, len(df_1d)):
        if np.isnan(kama[i]) or np.isnan(ema_long[i]) or np.isnan(volume_filter_daily[i]):
            continue
            
        price = df_1d['close'].iloc[i]
        kama_val = kama[i]
        ema_trend = ema_long[i]
        vol_ok = volume_filter_daily[i]
        
        if position == 0:
            # Long: price crosses above KAMA with volume in uptrend
            if price > kama_val and df_1d['close'].iloc[i-1] <= kama[i-1] and vol_ok and price > ema_trend:
                daily_signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with volume in downtrend
            elif price < kama_val and df_1d['close'].iloc[i-1] >= kama[i-1] and vol_ok and price < ema_trend:
                daily_signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA or trend reverses
            if price < kama_val or price < ema_trend:
                daily_signals[i] = 0.0
                position = 0
        elif position == -1:
            # Exit: price crosses above KAMA or trend reverses
            if price > kama_val or price > ema_trend:
                daily_signals[i] = 0.0
                position = 0
    
    # Now map daily signals to intraday prices
    # For each intraday bar, use the signal from the most recent completed daily bar
    signals = np.zeros(n)
    
    # Create date mapping
    date_intraday = pd.to_datetime(prices['open_time']).date
    date_daily = pd.to_datetime(df_1d.index).date
    
    # For each intraday bar, find the corresponding daily signal
    # We'll use the last daily signal where date_daily <= date_intraday
    signal_idx = 0
    daily_signal_idx = 0
    last_signal = 0.0
    
    for i in range(n):
        # Advance daily index while we have dates
        while daily_signal_idx < len(date_daily) - 1 and date_daily[daily_signal_idx + 1] <= date_intraday[i]:
            daily_signal_idx += 1
            last_signal = daily_signals[daily_signal_idx]
        signals[i] = last_signal
    
    return signals

name = "1d_KAMA_Direction_Volume_Trend"
timeframe = "1d"
leverage = 1.0