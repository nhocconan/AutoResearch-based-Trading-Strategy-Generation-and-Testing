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
    
    # Get 1d data for daily indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily RSI (14) - mean reversion signal
    delta = np.diff(df_1d['close'])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[0:14])
    avg_loss[13] = np.mean(loss[0:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # align with df_1d index
    
    # Daily ADX (14) - trend strength filter
    tr1 = np.abs(df_1d['high'] - df_1d['low'])
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    plus_dm = np.where((df_1d['high'] - np.roll(df_1d['high'], 1)) > (np.roll(df_1d['low'], 1) - df_1d['low']), 
                       np.maximum(df_1d['high'] - np.roll(df_1d['high'], 1), 0), 0)
    minus_dm = np.where((np.roll(df_1d['low'], 1) - df_1d['low']) > (df_1d['high'] - np.roll(df_1d['high'], 1)), 
                        np.maximum(np.roll(df_1d['low'], 1) - df_1d['low'], 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    plus_di = 100 * np.divide(
        np.convolve(plus_dm, np.ones(14)/14, mode='full')[:len(plus_dm)], 
        atr, 
        out=np.zeros_like(atr), 
        where=atr!=0
    )
    minus_di = 100 * np.divide(
        np.convolve(minus_dm, np.ones(14)/14, mode='full')[:len(minus_dm)], 
        atr, 
        out=np.zeros_like(atr), 
        where=atr!=0
    )
    dx = np.divide(np.abs(plus_di - minus_di), (plus_di + minus_di), out=np.zeros_like(plus_di), where=(plus_di + minus_di)!=0) * 100
    adx = np.convolve(dx, np.ones(14)/14, mode='full')[:len(dx)]
    adx = np.concatenate([np.full(27, np.nan), adx[27:]])  # align with df_1d index
    
    # Weekly EMA (20) - trend direction filter
    ema_20w = np.zeros_like(df_1w['close'])
    ema_20w[0] = df_1w['close'].iloc[0]
    alpha = 2 / (20 + 1)
    for i in range(1, len(ema_20w)):
        ema_20w[i] = alpha * df_1w['close'].iloc[i] + (1 - alpha) * ema_20w[i-1]
    
    # Align indicators to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(ema_20w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Mean reentry condition: RSI extreme + weak trend (ADX < 25)
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        weak_trend = adx_aligned[i] < 25
        
        # Trend filter: price above/below weekly EMA
        price_above_weekly_ema = close[i] > ema_20w_aligned[i]
        price_below_weekly_ema = close[i] < ema_20w_aligned[i]
        
        # Entry conditions
        long_entry = rsi_oversold and weak_trend and price_above_weekly_ema
        short_entry = rsi_overbought and weak_trend and price_below_weekly_ema
        
        # Exit conditions: RSI returns to neutral or trend strengthens
        exit_long = position == 1 and (rsi_aligned[i] > 50 or adx_aligned[i] > 30)
        exit_short = position == -1 and (rsi_aligned[i] < 50 or adx_aligned[i] > 30)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_rsi_adx_ema_mean_reversion"
timeframe = "1d"
leverage = 1.0