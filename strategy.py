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
    
    # Get 12h data for higher timeframe context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h ATR(14)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h ADX(14)
    plus_dm = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    minus_dm = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_14_12h + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_14_12h + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_14_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_14_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_14_12h)
    
    # Calculate 12h RSI(14)
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_12h = 100 - (100 / (1 + rs))
    rsi_14_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_14_12h)
    
    # Calculate 12h volume SMA(20)
    vol_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_20_12h)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_14_12h_aligned[i]) or 
            np.isnan(rsi_14_12h_aligned[i]) or
            np.isnan(vol_sma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_14_12h_aligned[i] > 25
        
        # Momentum filter: RSI not in extreme overbought/oversold
        rsi_not_extreme = (rsi_14_12h_aligned[i] > 30) and (rsi_14_12h_aligned[i] < 70)
        
        # Volume filter: above average volume
        vol_spike = volume[i] > vol_sma_20_12h_aligned[i]
        
        # Long conditions: strong trend + bullish momentum + volume spike
        long_condition = strong_trend and (close[i] > close[i-1]) and rsi_not_extreme and vol_spike
        
        # Short conditions: strong trend + bearish momentum + volume spike
        short_condition = strong_trend and (close[i] < close[i-1]) and rsi_not_extreme and vol_spike
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: loss of momentum or trend weakness
        elif position == 1 and (rsi_14_12h_aligned[i] >= 70 or adx_14_12h_aligned[i] < 20):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (rsi_14_12h_aligned[i] <= 30 or adx_14_12h_aligned[i] < 20):
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

name = "6h_ADX25_RSI14_VolumeFilter_Session"
timeframe = "6h"
leverage = 1.0