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
    open_time = prices['open_time'].values
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (10-period)
    highest_10 = pd.Series(df_12h['high'].values).rolling(window=10, min_periods=10).max().values
    lowest_10 = pd.Series(df_12h['low'].values).rolling(window=10, min_periods=10).min().values
    donchian_high_12h = align_htf_to_ltf(prices, df_12h, highest_10, additional_delay_bars=0)
    donchian_low_12h = align_htf_to_ltf(prices, df_12h, lowest_10, additional_delay_bars=0)
    
    # Calculate 12h ATR for volatility filter
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    prev_close = np.concatenate([[close_12h[0]], close_12h[:-1]])
    tr = np.maximum(high_12h - low_12h,
                    np.maximum(np.abs(high_12h - prev_close),
                               np.abs(low_12h - prev_close)))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_12h).rolling(window=50, min_periods=50).mean().values
    volatility_ratio_12h = atr_12h / (atr_ma_50 + 1e-10)
    volatility_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, volatility_ratio_12h, additional_delay_bars=0)
    
    # Calculate 12h RSI for momentum filter
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h, additional_delay_bars=0)
    
    # Calculate 12h EMA for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h, additional_delay_bars=0)
    
    # Calculate 12h volume ratio
    vol_ma_20_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_ratio_12h = df_12h['volume'].values / (vol_ma_20_12h + 1e-10)
    volume_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ratio_12h, additional_delay_bars=0)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or
            np.isnan(volume_ratio_12h_aligned[i]) or np.isnan(volatility_ratio_12h_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Entry conditions for 12h timeframe
        # Long: Donchian breakout + volume + trend + momentum + volatility filter
        # Short: Donchian breakdown + volume + trend + momentum + volatility filter
        
        # Long conditions
        if (close[i] > donchian_high_12h[i] and     # Donchian breakout
            close[i] > ema_50_12h_aligned[i] and    # Above 12h EMA50 (uptrend)
            rsi_12h_aligned[i] > 50 and             # Bullish momentum
            volume_ratio_12h_aligned[i] > 1.3 and   # Volume confirmation
            volatility_ratio_12h_aligned[i] > 0.7): # Avoid extremely low volatility
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < donchian_low_12h[i] and      # Donchian breakdown
              close[i] < ema_50_12h_aligned[i] and    # Below 12h EMA50 (downtrend)
              rsi_12h_aligned[i] < 50 and             # Bearish momentum
              volume_ratio_12h_aligned[i] > 1.3 and   # Volume confirmation
              volatility_ratio_12h_aligned[i] > 0.7): # Avoid extremely low volatility
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_Volume_Trend_Momentum"
timeframe = "12h"
leverage = 1.0