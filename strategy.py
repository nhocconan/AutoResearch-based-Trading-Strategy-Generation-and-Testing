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
    
    # Calculate daily EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4h data for price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-day RSI on 1D timeframe
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=4, min_periods=4).mean()
    avg_loss = loss.rolling(window=4, min_periods=4).mean()
    rs = avg_gain / avg_loss
    rsi_4d = 100 - (100 / (1 + rs))
    rsi_4d_aligned = align_htf_to_ltf(prices, df_1d, rsi_4d.values)
    
    # Calculate 4h Bollinger Bands (20, 2.0)
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb = ma_20 + (std_20 * 2.0)
    lower_bb = ma_20 - (std_20 * 2.0)
    
    # Calculate 4h volume moving average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_4d_aligned[i]) or
            np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: current volume above average
        volume_filter = volume[i] > vol_ma_20[i] * 1.5
        
        # Bollinger Band conditions
        near_upper_bb = close[i] > upper_bb[i] * 0.99
        near_lower_bb = close[i] < lower_bb[i] * 1.01
        
        # RSI conditions for momentum
        rsi_overbought = rsi_4d_aligned[i] > 70
        rsi_oversold = rsi_4d_aligned[i] < 30
        
        # Long conditions: uptrend + volume + RSI not overbought + near lower BB (mean reversion)
        long_condition = (price_above_ema and 
                         volume_filter and 
                         not rsi_overbought and 
                         near_lower_bb)
        
        # Short conditions: downtrend + volume + RSI not oversold + near upper BB (mean reversion)
        short_condition = (price_below_ema and 
                          volume_filter and 
                          not rsi_oversold and 
                          near_upper_bb)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or RSI extreme
        elif position == 1 and (not price_above_ema or rsi_4d_aligned[i] > 80):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or rsi_4d_aligned[i] < 20):
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

name = "1d_EMA50_4dRSI_4hBB_MeanReversion"
timeframe = "4h"
leverage = 1.0