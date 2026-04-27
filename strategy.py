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
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA 34 for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d RSI 14
    delta_1d = pd.Series(close_1d).diff()
    gain_1d = delta_1d.where(delta_1d > 0, 0)
    loss_1d = -delta_1d.where(delta_1d < 0, 0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d ATR 14
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.abs(high_1d[1:] - close_1d[:-1]), 
                       np.abs(low_1d[1:] - close_1d[:-1]))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d volume moving average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 6h Donchian channels (20 periods)
    donchian_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donchian_len - 1, n):
        upper[i] = np.max(high[i-donchian_len+1:i+1])
        lower[i] = np.min(low[i-donchian_len+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(upper[i]) or 
            np.isnan(lower[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # RSI filter: avoid extreme overbought/oversold conditions
        rsi_not_overbought = rsi_1d_aligned[i] < 70
        rsi_not_oversold = rsi_1d_aligned[i] > 30
        
        # Volatility filter: only trade when volatility is reasonable
        vol_filter = atr_1d_aligned[i] > 0
        
        # Volume filter: current volume above 1d average
        volume_filter = volume[i] > vol_ma_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper[i-1] if i > 0 else False
        breakout_down = close[i] < lower[i-1] if i > 0 else False
        
        # Long conditions: price above EMA34 + Donchian breakout up + volume + volatility
        long_condition = (price_above_ema and 
                         breakout_up and 
                         rsi_not_overbought and 
                         volume_filter and 
                         vol_filter)
        
        # Short conditions: price below EMA34 + Donchian breakout down + volume + volatility
        short_condition = (price_below_ema and 
                          breakout_down and 
                          rsi_not_oversold and 
                          volume_filter and 
                          vol_filter)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or Donchian opposite breakout
        elif position == 1 and (not price_above_ema or breakout_down):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or breakout_up):
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

name = "6h_Donchian20_Breakout_1dEMA34_RSI14_VolumeFilter"
timeframe = "6h"
leverage = 1.0