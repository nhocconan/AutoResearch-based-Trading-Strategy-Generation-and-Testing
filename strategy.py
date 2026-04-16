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
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # 6h Donchian channels (20 periods)
    high_20_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_20_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_upper_6h = align_htf_to_ltf(prices, df_6h, high_20_6h)
    donchian_lower_6h = align_htf_to_ltf(prices, df_6h, low_20_6h)
    
    # 6h EMA34 for trend filter
    close_6h_series = pd.Series(close_6h)
    ema_34_6h = close_6h_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_34_6h)
    
    # === 1d data (HTF for regime) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for long-term trend
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 6h RSI for momentum ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume spike detection ===
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_ratio = volume / vol_ma_10
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_6h[i]) or np.isnan(donchian_lower_6h[i]) or 
            np.isnan(ema_34_6h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_6h = donchian_upper_6h[i]
        lower_6h = donchian_lower_6h[i]
        ema_34_6h_val = ema_34_6h_aligned[i]
        ema_200_1d_val = ema_200_1d_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower OR RSI becomes overbought
            if (price < lower_6h) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper OR RSI becomes oversold
            if (price > upper_6h) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND above EMA34 (trend filter) 
            # AND price above 1d EMA200 (bullish regime) AND RSI not overbought AND volume spike
            if (price > upper_6h) and (price > ema_34_6h_val) and (price > ema_200_1d_val) and \
               (rsi_val < 60) and (vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below Donchian lower AND below EMA34 (trend filter) 
            # AND price below 1d EMA200 (bearish regime) AND RSI not oversold AND volume spike
            elif (price < lower_6h) and (price < ema_34_6h_val) and (price < ema_200_1d_val) and \
                 (rsi_val > 40) and (vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_EMA34_EMA200_RSI_Volume"
timeframe = "6h"
leverage = 1.0