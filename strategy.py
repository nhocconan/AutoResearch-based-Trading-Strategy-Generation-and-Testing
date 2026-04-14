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
    
    # Load 12h data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h ATR for volatility filter (14-period)
    high_low = high_12h - low_12h
    high_close = np.abs(high_12h - np.roll(close_12h, 1))
    low_close = np.abs(low_12h - np.roll(close_12h, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr_12h = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_12h_series = pd.Series(tr_12h)
    atr_12h = tr_12h_series.rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 4h Donchian channels (20-period) - breakout levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 12h RSI (14-period) for momentum filter
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi_14_12h = 100 - (100 / (1 + rs))
    rsi_14_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_14_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(atr_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(rsi_14_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            continue
        
        # Trend filter: 12h EMA50 slope (rising/falling)
        ema_slope = ema50_12h_aligned[i] - ema50_12h_aligned[i-1] if i > 0 else 0
        
        # Momentum filter: 12h RSI not extreme
        rsi_momentum = (rsi_14_12h_aligned[i] > 30) and (rsi_14_12h_aligned[i] < 70)
        
        # Volatility filter: current 12h ATR > 30-day average ATR
        vol_filter = True
        if i >= 240:  # 30 days of 12h bars
            atr_avg = np.mean(atr_12h_aligned[max(0, i-240):i])
            vol_filter = atr_12h_aligned[i] > atr_avg * 0.8
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high with 12h trend up and momentum
            if (close[i] > donchian_high[i] and 
                ema_slope > 0 and 
                rsi_momentum and 
                vol_filter):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 4h Donchian low with 12h trend down and momentum
            elif (close[i] < donchian_low[i] and 
                  ema_slope < 0 and 
                  rsi_momentum and 
                  vol_filter):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price breaks below 4h Donchian low or 12h trend turns down
            if close[i] < donchian_low[i] or ema_slope < 0:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price breaks above 4h Donchian high or 12h trend turns up
            if close[i] > donchian_high[i] or ema_slope > 0:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_12h_EMA50_RSI_Donchian_Breakout"
timeframe = "4h"
leverage = 1.0