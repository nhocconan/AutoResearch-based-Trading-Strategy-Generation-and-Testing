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
    
    # Get daily data for trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA(21) for trend
    close_1d_series = pd.Series(close_1d)
    ema_21_1d = close_1d_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate daily ATR(14) for volatility
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Calculate daily volume moving average
    vol_s = pd.Series(volume_1d)
    vol_ma_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 12h ATR(14) for position sizing and volatility
    tr1_h = np.abs(high - low)
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr1_h[0] = tr2_h[0] = tr3_h[0] = np.nan
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr_12h = np.full(n, np.nan)
    for i in range(14, n):
        atr_12h[i] = np.mean(tr_h[i-14:i+1])
    
    # Calculate 12h volume moving average
    vol_s_h = pd.Series(volume)
    vol_ma_20_h = vol_s_h.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr_12h[i]) or np.isnan(vol_ma_20_h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5 * ATR MA(10) to avoid low volatility
        atr_ma_10 = np.full(n, np.nan)
        for j in range(23, n):  # 14 + 9 for 10-period MA
            if not np.isnan(np.mean(atr_12h[j-9:j+1])):
                atr_ma_10[j] = np.mean(atr_12h[j-9:j+1])
        vol_filter = atr_12h[i] > 0.5 * atr_ma_10[i] if not np.isnan(atr_ma_10[i]) else False
        
        # Volume filter: volume > 1.5 * 20-period MA
        vol_spike = volume[i] > 1.5 * vol_ma_20_h[i]
        
        # Trend filter: price relative to daily EMA21
        uptrend = close[i] > ema_21_1d_aligned[i]
        downtrend = close[i] < ema_21_1d_aligned[i]
        
        # Entry conditions: price above/below daily EMA with volatility and volume filters
        long_entry = uptrend and vol_filter and vol_spike
        short_entry = downtrend and vol_filter and vol_spike
        
        # Exit conditions: price crosses back to daily EMA21
        long_exit = close[i] < ema_21_1d_aligned[i]
        short_exit = close[i] > ema_21_1d_aligned[i]
        
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

name = "12h_1d_ema21_trend_vol_vol_filter"
timeframe = "12h"
leverage = 1.0