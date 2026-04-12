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
    
    # Get 1d data for context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA(50)
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily ATR(14)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d)
    delta = np.insert(delta, 0, np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(len(delta), np.nan)
    avg_loss = np.full(len(delta), np.nan)
    for i in range(14, len(delta)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align daily indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6h ATR(14) for position sizing
    tr1_h = np.abs(high - low)
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr1_h[0] = tr2_h[0] = tr3_h[0] = np.nan
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr_6h = np.full(n, np.nan)
    for i in range(14, n):
        atr_6h[i] = np.mean(tr_h[i-14:i+1])
    
    # Calculate 6h volume moving average
    vol_s = pd.Series(volume)
    vol_ma_20_6h = vol_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_6h[i]) or np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 6h volume > 1.5 * 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20_6h[i]
        
        # Volatility filter: daily ATR > 0.5 * its 20-period MA (avoid low volatility)
        atr_ma_20_1d = np.full(len(df_1d), np.nan)
        for j in range(34, len(df_1d)):  # 14 + 19 for 20-period MA
            if not np.isnan(np.mean(atr_1d[j-19:j+1])):
                atr_ma_20_1d[j] = np.mean(atr_1d[j-19:j+1])
        atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
        vol_filter_daily = atr_1d_aligned[i] > 0.5 * atr_ma_20_1d_aligned[i] if not np.isnan(atr_ma_20_1d_aligned[i]) else False
        
        # Trend filter: price above/below daily EMA50
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_not_overbought = rsi_1d_aligned[i] < 70
        rsi_not_oversold = rsi_1d_aligned[i] > 30
        
        # Entry conditions: trend + volume + volatility + RSI filter
        long_entry = above_ema and vol_filter and vol_filter_daily and rsi_not_overbought
        short_entry = below_ema and vol_filter and vol_filter_daily and rsi_not_oversold
        
        # Exit conditions: trend reversal or RSI extreme
        long_exit = below_ema or rsi_1d_aligned[i] >= 70
        short_exit = above_ema or rsi_1d_aligned[i] <= 30
        
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

name = "6h_1d_ema50_rsi_vol_vol_filter_v1"
timeframe = "6h"
leverage = 1.0