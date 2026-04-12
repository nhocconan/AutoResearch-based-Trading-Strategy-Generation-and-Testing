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
    
    # Get weekly data for regime filter (ADX)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily data for KAMA trend and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily close (trend filter)
    # Efficiency ratio
    change = np.abs(np.diff(df_1d['close'], n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(df_1d['close'], n=1)), axis=0)  # 10-period volatility
    # Handle first 10 values
    change_full = np.full(len(df_1d['close']), np.nan)
    volatility_full = np.full(len(df_1d['close']), np.nan)
    change_full[10:] = change
    volatility_full[10:] = volatility
    er = np.where(volatility_full != 0, change_full / volatility_full, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full(len(df_1d['close']), np.nan)
    kama[9] = df_1d['close'].iloc[9] if hasattr(df_1d['close'], 'iloc') else df_1d['close'][9]
    for i in range(10, len(df_1d['close'])):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] if hasattr(df_1d['close'], 'iloc') else df_1d['close'][i] - kama[i-1])
    
    # Calculate RSI on daily close
    delta = np.diff(df_1d['close'])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average
    avg_gain = np.full(len(df_1d['close']), np.nan)
    avg_loss = np.full(len(df_1d['close']), np.nan)
    avg_gain[14] = np.mean(gain[1:15]) if len(gain) >= 15 else np.nan
    avg_loss[14] = np.mean(loss[1:15]) if len(loss) >= 15 else np.nan
    # Wilder smoothing
    for i in range(15, len(df_1d['close'])):
        if not np.isnan(avg_gain[i-1]):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align KAMA and RSI to daily timeframe (already aligned as daily)
    kama_aligned = kama  # already daily
    rsi_aligned = rsi    # already daily
    
    # Calculate ADX on weekly for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1w)
    down_move = -np.diff(low_1w)  # negative of negative = positive
    up_move = np.insert(up_move, 0, np.nan)
    down_move = np.insert(down_move, 0, np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) < period:
            return result
        first_avg = np.nansum(x[1:period+1])
        result[period] = first_avg
        for i in range(period+1, len(x)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + x[i]) / period
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    plus_di_1w = 100 * wilders_smoothing(plus_dm, 14) / atr_1w
    minus_di_1w = 100 * wilders_smoothing(minus_dm, 14) / atr_1w
    dx = np.where((plus_di_1w + minus_di_1w) != 0, 
                  100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w), 0)
    adx_1w = wilders_smoothing(dx, 14)
    
    # Align ADX to daily timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume filter: 20-day EMA on daily volume
    # Get daily volume
    vol_1d = df_1d['volume'].values
    vol_ema_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current daily volume > 1.5x EMA
        # Need to get daily volume aligned
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        volume_filter = vol_1d_aligned[i] > vol_ema_aligned[i] * 1.5
        
        # Regime filter: ADX > 25 for trending market
        regime_filter = adx_1w_aligned[i] > 25
        
        # Entry conditions: 
        # Long: price > KAMA (uptrend) AND RSI < 40 (pullback) AND volume spike AND trending
        # Short: price < KAMA (downtrend) AND RSI > 60 (bounce) AND volume spike AND trending
        price_position = close[i]  # current price
        kama_level = kama_aligned[i]
        rsi_level = rsi_aligned[i]
        
        long_entry = (price_position > kama_level) and (rsi_level < 40) and volume_filter and regime_filter
        short_entry = (price_position < kama_level) and (rsi_level > 60) and volume_filter and regime_filter
        
        # Exit conditions: 
        # Long exit: RSI > 60 (overbought) or price < KAMA (trend change)
        # Short exit: RSI < 40 (oversold) or price > KAMA (trend change)
        long_exit = (rsi_level > 60) or (price_position < kama_level)
        short_exit = (rsi_level < 40) or (price_position > kama_level)
        
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

name = "1d_kama_rsi_adx_volume_filter_v1"
timeframe = "1d"
leverage = 1.0