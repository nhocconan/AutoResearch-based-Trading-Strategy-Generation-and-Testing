#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Get daily data for KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily KAMA calculation
    daily_close = df_1d['close'].values
    change = np.abs(np.diff(daily_close, 1))
    change = np.insert(change, 0, 0)
    volatility = np.zeros_like(daily_close)
    for i in range(1, len(daily_close)):
        volatility[i] = volatility[i-1] + change[i] - (change[i-10] if i >= 10 else 0)
    er = np.zeros_like(daily_close)
    for i in range(len(daily_close)):
        if volatility[i] != 0:
            er[i] = np.abs(daily_close[i] - daily_close[i-9]) / volatility[i] if i >= 9 else 0
        else:
            er[i] = 0
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(daily_close)
    kama[0] = daily_close[0]
    for i in range(1, len(daily_close)):
        kama[i] = kama[i-1] + sc[i] * (daily_close[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Daily RSI
    delta = np.diff(daily_close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Daily volume average
    daily_volume = df_1d['volume'].values
    vol_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    # ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-14:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(weekly_ema_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA
        above_weekly_ema = close[i] > weekly_ema_aligned[i]
        below_weekly_ema = close[i] < weekly_ema_aligned[i]
        
        # KAMA direction: price above/below KAMA
        above_kama = close[i] > kama_aligned[i]
        below_kama = close[i] < kama_aligned[i]
        
        # RSI filter: avoid extreme overbought/oversold
        rsi_not_extreme = (rsi_aligned[i] > 20) and (rsi_aligned[i] < 80)
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # Volatility filter: ATR > 0.5 * 20-period ATR mean
        atr_ma = np.full(n, np.nan)
        if i >= 34:
            atr_ma[i] = np.nanmean(atr[i-20:i])
        vol_filter = atr[i] > atr_ma[i] * 0.5 if not np.isnan(atr_ma[i]) else True
        
        # Entry conditions: KAMA direction with weekly trend filter
        long_entry = above_kama and above_weekly_ema and rsi_not_extreme and volume_filter and vol_filter
        short_entry = below_kama and below_weekly_ema and rsi_not_extreme and volume_filter and vol_filter
        
        # Exit conditions: opposite KAMA cross
        long_exit = below_kama
        short_exit = above_kama
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_kama_rsi_volume_filter_v1"
timeframe = "1d"
leverage = 1.0