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
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily 14-period RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    alpha = 1/14
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=alpha, adjust=False, min_periods=14).mean().values
    avg_loss = loss_series.ewm(alpha=alpha, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    
    # Calculate daily 20-period EMA
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily 14-period ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Calculate 4h ATR for volatility filter
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = high[0] - low[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or 
            np.isnan(atr_aligned[i]) or
            np.isnan(atr_4h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility
        vol_ok = atr_4h[i] > (atr_4h[i] * 0.2)  # Always true, keeping for structure
        
        # Volume filter: require above average volume
        vol_filter = volume[i] > vol_ma_20[i]
        
        # Trend filter: price above/below daily EMA20
        uptrend = close[i] > ema_20_aligned[i]
        downtrend = close[i] < ema_20_aligned[i]
        
        # RSI conditions: extreme levels for mean reversion
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        # Long conditions: uptrend + oversold + volume filter
        long_condition = uptrend and rsi_oversold and vol_filter
        
        # Short conditions: downtrend + overbought + volume filter
        short_condition = downtrend and rsi_overbought and vol_filter
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: RSI returns to neutral zone
        elif position == 1 and rsi_aligned[i] > 50:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi_aligned[i] < 50:
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

name = "4h_DailyRSI_EMA20_MeanReversion"
timeframe = "4h"
leverage = 1.0