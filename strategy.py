#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate 12h EMA 50 for trend direction
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h RSI for overbought/oversold conditions
    delta_12h = pd.Series(close_12h).diff()
    gain_12h = delta_12h.where(delta_12h > 0, 0)
    loss_12h = -delta_12h.where(delta_12h < 0, 0)
    avg_gain_12h = gain_12h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_12h = loss_12h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_12h = avg_gain_12h / avg_loss_12h
    rsi_12h = 100 - (100 / (1 + rs_12h))
    rsi_12h = rsi_12h.values
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate 12h ATR for volatility filter
    tr_12h = np.maximum(high_12h[1:] - low_12h[1:], 
                        np.abs(high_12h[1:] - close_12h[:-1]), 
                        np.abs(low_12h[1:] - close_12h[:-1]))
    tr_12h = np.concatenate([[np.nan], tr_12h])
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 12h volume moving average
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate 4h Donchian channels for breakout signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper and lower bands (20-period)
    donch_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or
            np.isnan(donch_high_4h_aligned[i]) or
            np.isnan(donch_low_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        # RSI filter: avoid extreme overbought/oversold conditions
        rsi_not_overbought = rsi_12h_aligned[i] < 70
        rsi_not_oversold = rsi_12h_aligned[i] > 30
        
        # Volatility filter: only trade when volatility is reasonable
        vol_filter = atr_12h_aligned[i] > 0
        
        # Volume filter: current volume above 12h average
        volume_filter = volume[i] > vol_ma_12h_aligned[i]
        
        # Breakout filters: price breaking above/below 4h Donchian channels
        breakout_up = close[i] > donch_high_4h_aligned[i]
        breakout_down = close[i] < donch_low_4h_aligned[i]
        
        # Long conditions: price above EMA50 + RSI not overbought + volume + volatility + breakout up
        long_condition = (price_above_ema and 
                         rsi_not_overbought and 
                         volume_filter and 
                         vol_filter and 
                         breakout_up)
        
        # Short conditions: price below EMA50 + RSI not oversold + volume + volatility + breakout down
        short_condition = (price_below_ema and 
                          rsi_not_oversold and 
                          volume_filter and 
                          vol_filter and 
                          breakout_down)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or breakout failure
        elif position == 1 and (not price_above_ema or not breakout_up):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or not breakout_down):
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

name = "4h_Donchian_Breakout_12hEMA50_RSI_Volume_Filter"
timeframe = "4h"
leverage = 1.0