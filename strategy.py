#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 200-day EMA for long-term trend
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 50-day EMA for medium-term trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 14-day RSI for momentum
    delta_1d = pd.Series(close_1d).diff()
    gain_1d = delta_1d.where(delta_1d > 0, 0)
    loss_1d = -delta_1d.where(delta_1d < 0, 0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 14-day ATR for volatility
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.abs(high_1d[1:] - close_1d[:-1]), 
                       np.abs(low_1d[1:] - close_1d[:-1]))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 20-day volume moving average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 200-day EMA
        price_above_ema200 = close[i] > ema_200_1d_aligned[i]
        price_below_ema200 = close[i] < ema_200_1d_aligned[i]
        
        # Medium-term trend: 50-day EMA above/below 200-day EMA
        ema50_above_ema200 = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
        ema50_below_ema200 = ema_50_1d_aligned[i] < ema_200_1d_aligned[i]
        
        # RSI filter: avoid extreme conditions
        rsi_not_overbought = rsi_1d_aligned[i] < 70
        rsi_not_oversold = rsi_1d_aligned[i] > 30
        
        # Volume filter: current volume above daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i]
        
        # Long conditions: bullish alignment + volume
        long_condition = (price_above_ema200 and 
                         ema50_above_ema200 and 
                         rsi_not_overbought and 
                         volume_filter)
        
        # Short conditions: bearish alignment + volume
        short_condition = (price_below_ema200 and 
                          ema50_below_ema200 and 
                          rsi_not_oversold and 
                          volume_filter)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal
        elif position == 1 and not (price_above_ema200 and ema50_above_ema200):
            signals[i] = 0.0
            position = 0
        elif position == -1 and not (price_below_ema200 and ema50_below_ema200):
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

name = "1d_EMA200_EMA50_RSI14_VolumeFilter"
timeframe = "1d"
leverage = 1.0