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
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian(15) for entry/exit levels
    high_15_12h = pd.Series(high_12h).rolling(window=15, min_periods=15).max().values
    low_15_12h = pd.Series(low_12h).rolling(window=15, min_periods=15).min().values
    donchian_upper_12h = align_htf_to_ltf(prices, df_12h, high_15_12h)
    donchian_lower_12h = align_htf_to_ltf(prices, df_12h, low_15_12h)
    
    # === 1d data (HTF for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d data (HTF for volatility regime) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d ATR for volatility regime
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1d data (HTF for volume confirmation) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        upper_12h = donchian_upper_12h[i]
        lower_12h = donchian_lower_12h[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        vol_ma_20_1d_val = vol_ma_20_1d_aligned[i]
        volume_current = volume[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower OR trend reverses
            if (price < lower_12h) or (price < ema_50_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper OR trend reverses
            if (price > upper_12h) or (price > ema_50_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # Volume confirmation: current volume > 1.5x 20-day average
                vol_confirm = volume_current > (1.5 * vol_ma_20_1d_val)
                
                # Volatility filter: avoid extremely high volatility periods
                vol_filter = atr_1d_val < np.percentile(atr_1d_aligned[:i+1], 80)
                
                # LONG: Price breaks above Donchian upper AND above EMA50 (trend filter) 
                # AND volume confirmation AND volatility filter
                if (price > upper_12h) and (price > ema_50_1d_val) and \
                   vol_confirm and vol_filter:
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below Donchian lower AND below EMA50 (trend filter) 
                # AND volume confirmation AND volatility filter
                elif (price < lower_12h) and (price < ema_50_1d_val) and \
                     vol_confirm and vol_filter:
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

name = "12h_Donchian_Breakout_EMA50_Volume_Filter"
timeframe = "12h"
leverage = 1.0