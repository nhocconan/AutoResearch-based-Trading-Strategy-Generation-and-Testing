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
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Get 4h data for entry timing
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4-hour RSI for mean reversion signals
    delta = pd.Series(close_1d).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate 4-hour Bollinger Bands for volatility context
    sma_20_4h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20_4h = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20_4h + 2 * std_20_4h
    bb_lower = sma_20_4h - 2 * std_20_4h
    
    # Calculate 4-hour volume moving average for confirmation
    vol_ma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(sma_20_4h[i]) or
            np.isnan(std_20_4h[i]) or
            np.isnan(vol_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Volatility filter: avoid extreme volatility periods
        atr_threshold = np.nanpercentile(atr_14_1d_aligned[max(0, i-100):i+1], 80) if i >= 50 else atr_14_1d_aligned[i]
        low_volatility = atr_14_1d_aligned[i] < atr_threshold
        
        # Volume filter: current volume above average
        volume_filter = vol_ma_20_4h[i] > 0 and volume[i] > vol_ma_20_4h[i] * 1.2
        
        # Mean reversion signals: RSI extremes with Bollinger Band touch
        rsi_oversold = rsi_14_1d_aligned[i] < 30
        rsi_overbought = rsi_14_1d_aligned[i] > 70
        bb_touch_lower = close[i] <= bb_lower[i]
        bb_touch_upper = close[i] >= bb_upper[i]
        
        # Long conditions: oversold RSI + BB lower touch + volume + low volatility
        long_condition = (rsi_oversold and 
                         bb_touch_lower and 
                         volume_filter and 
                         low_volatility)
        
        # Short conditions: overbought RSI + BB upper touch + volume + low volatility
        short_condition = (rsi_overbought and 
                          bb_touch_upper and 
                          volume_filter and 
                          low_volatility)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: RSI returns to neutral territory
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

name = "4h_RSI_Bollinger_MeanReversion_EMA34Trend"
timeframe = "4h"
leverage = 1.0