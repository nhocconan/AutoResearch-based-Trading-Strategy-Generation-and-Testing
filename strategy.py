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
    
    # Get daily data for trend and volatility context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA(50) for trend
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily ATR(14) for volatility
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Align daily indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h ATR(14) for position sizing
    tr1_h = np.abs(high - low)
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr1_h[0] = tr2_h[0] = tr3_h[0] = np.nan
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr_4h = np.full(n, np.nan)
    for i in range(14, n):
        atr_4h[i] = np.mean(tr_h[i-14:i+1])
    
    # Calculate 4h EMA(21) for trend
    close_s = pd.Series(close)
    ema_21_4h = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 4h volume moving average
    vol_s = pd.Series(volume)
    vol_ma_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema_21_4h[i]) or np.isnan(atr_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5 * ATR MA(10) to avoid low volatility
        atr_ma_10 = np.full(n, np.nan)
        for j in range(23, n):  # 14 + 9 for 10-period MA
            if not np.isnan(np.mean(atr_1d_aligned[j-9:j+1])):
                atr_ma_10[j] = np.mean(atr_1d_aligned[j-9:j+1])
        vol_filter = atr_1d_aligned[i] > 0.5 * atr_ma_10[i] if not np.isnan(atr_ma_10[i]) else False
        
        # Volume filter: volume > 1.5 * 20-period MA
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price relative to daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Trend filter: 4h EMA21
        price_above_ema = close[i] > ema_21_4h[i]
        price_below_ema = close[i] < ema_21_4h[i]
        
        # Entry conditions: price above/below both EMAs with volatility and volume filters
        long_entry = price_above_ema and uptrend and vol_filter and vol_spike
        short_entry = price_below_ema and downtrend and vol_filter and vol_spike
        
        # Exit conditions: price crosses back to EMA21
        long_exit = close[i] < ema_21_4h[i]
        short_exit = close[i] > ema_21_4h[i]
        
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

name = "4h_1d_ema_dual_trend_filter_vol_vol"
timeframe = "4h"
leverage = 1.0