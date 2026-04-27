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
    
    # Get daily data for higher timeframe context (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily RSI(14) for momentum
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d = rsi_14_1d.fillna(50).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4h Bollinger Bands (20,2) for mean reversion signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    sma_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bb_upper_20 = sma_20_4h + 2 * std_20_4h
    bb_lower_20 = sma_20_4h - 2 * std_20_4h
    bb_upper_aligned = align_htf_to_ltf(prices, df_4h, bb_upper_20)
    bb_lower_aligned = align_htf_to_ltf(prices, df_4h, bb_lower_20)
    
    # Calculate 4h volume moving average for confirmation
    vol_ma_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR above 20-period average
        vol_filter = atr_14_1d_aligned[i] > 0
        
        # Mean reversion signals: price touches Bollinger Bands
        bb_touch_upper = close[i] >= bb_upper_aligned[i]
        bb_touch_lower = close[i] <= bb_lower_aligned[i]
        
        # RSI conditions for momentum confirmation
        rsi_oversold = rsi_14_1d_aligned[i] < 30
        rsi_overbought = rsi_14_1d_aligned[i] > 70
        
        # Trend filter: price relative to daily EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: current 4h volume above average
        volume_filter = volume[i] > vol_ma_4h_aligned[i] * 1.2
        
        # Long conditions: oversold RSI + lower BB touch + volume + uptrend bias
        long_condition = (rsi_oversold and 
                         bb_touch_lower and 
                         volume_filter and
                         price_above_ema)
        
        # Short conditions: overbought RSI + upper BB touch + volume + downtrend bias
        short_condition = (rsi_overbought and 
                          bb_touch_upper and 
                          volume_filter and
                          price_below_ema)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: RSI returns to neutral zone
        elif position == 1 and rsi_14_1d_aligned[i] > 50:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi_14_1d_aligned[i] < 50:
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

name = "1d_EMA50_RSI14_4hBBands_MeanReversion"
timeframe = "4h"
leverage = 1.0