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
    
    # === 1d data (HTF for 12h strategy) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Donchian(20) for entry/exit levels
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper_1d = align_htf_to_ltf(prices, df_1d, high_20_1d)
    donchian_lower_1d = align_htf_to_ltf(prices, df_1d, low_20_1d)
    
    # 1d ATR for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h EMA34 for additional trend filter
    close_12h_series = pd.Series(close)
    ema_34_12h = close_12h_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # 12h volume spike detection
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_ratio = volume / vol_ma_10
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_1d[i]) or np.isnan(donchian_lower_1d[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_1d = donchian_upper_1d[i]
        lower_1d = donchian_lower_1d[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        ema_34_12h_val = ema_34_12h[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower OR EMA34 trend breaks
            if (price < lower_1d) or (price < ema_34_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper OR EMA34 trend breaks
            if (price > upper_1d) or (price > ema_34_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND above daily EMA34 (trend filter) 
            # AND 12h EMA34 supports trend AND volume spike AND volatility not extreme
            if (price > upper_1d) and (price > ema_34_1d_val) and (ema_34_12h_val > ema_34_1d_val) and \
               (vol_ratio_val > 2.0) and (atr_1d_val < np.percentile(atr_1d_aligned[:i+1], 80)):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below Donchian lower AND below daily EMA34 (trend filter) 
            # AND 12h EMA34 supports trend AND volume spike AND volatility not extreme
            elif (price < lower_1d) and (price < ema_34_1d_val) and (ema_34_12h_val < ema_34_1d_val) and \
                 (vol_ratio_val > 2.0) and (atr_1d_val < np.percentile(atr_1d_aligned[:i+1], 80)):
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

name = "12h_Donchian_Breakout_EMA34_DualTrend_Volume"
timeframe = "12h"
leverage = 1.0