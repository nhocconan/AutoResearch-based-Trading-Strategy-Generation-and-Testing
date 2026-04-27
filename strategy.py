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
    
    # Get weekly data for EMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34)
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Get daily data for RSI and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    close_daily = df_daily['close'].values
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi)
    
    # Calculate daily volume MA(20)
    vol_daily = df_daily['volume'].values
    vol_ma_20_daily = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need weekly EMA, daily RSI, and daily volume MA
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_weekly_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema_weekly_val = ema_weekly_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_daily_aligned[i]
        
        # Volume filter: volume > 1.2x daily MA
        vol_filter = vol_now > 1.2 * vol_ma
        
        # Entry conditions
        if position == 0:
            # Long: price above weekly EMA, RSI > 50, volume breakout
            if close[i] > ema_weekly_val and rsi_val > 50 and vol_filter:
                signals[i] = size
                position = 1
            # Short: price below weekly EMA, RSI < 50, volume breakout
            elif close[i] < ema_weekly_val and rsi_val < 50 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly EMA or RSI < 40
            if close[i] < ema_weekly_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly EMA or RSI > 60
            if close[i] > ema_weekly_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyEMA34_RSI14_VolumeFilter"
timeframe = "1d"
leverage = 1.0