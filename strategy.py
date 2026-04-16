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
    
    # === 12h data (primary) ===
    # Donchian breakout with volume confirmation and trend filter
    
    # === 1w data (HTF for trend) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 12h indicators ===
    # Donchian channels (20 periods)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_20
    donchian_lower = low_20
    
    # EMA20 for trend confirmation
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # RSI for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_20[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        ema_20_val = ema_20[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        rsi_val = rsi[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower OR RSI becomes overbought
            if price < lower or rsi_val > 70:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper OR RSI becomes oversold
            if price > upper or rsi_val < 30:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND above EMA20 (trend filter) 
            # AND above weekly EMA34 (long-term trend) AND RSI not overbought AND volume spike
            if (price > upper) and (price > ema_20_val) and (price > ema_34_1w_val) and \
               (rsi_val < 60) and (vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below Donchian lower AND below EMA20 (trend filter) 
            # AND below weekly EMA34 (long-term trend) AND RSI not oversold AND volume spike
            elif (price < lower) and (price < ema_20_val) and (price < ema_34_1w_val) and \
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

name = "12h_Donchian_Breakout_EMA20_EMA34_1w_Volume"
timeframe = "12h"
leverage = 1.0