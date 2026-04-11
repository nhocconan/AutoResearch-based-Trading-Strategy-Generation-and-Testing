#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate CCI on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Price
    tp = (high_1d + low_1d + close_1d) / 3.0
    
    # SMA of TP
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    
    # Mean Deviation
    mad = np.abs(tp - sma_tp)
    mean_dev = pd.Series(mad).rolling(window=20, min_periods=20).mean().values
    
    # CCI
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    cci = np.where(mean_dev == 0, 0, cci)
    
    # Align CCI to 4h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    
    # Calculate daily ATR for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Load 1-hour data for volume confirmation
    df_1h = get_htf_data(prices, '1h')
    
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # Hourly volume confirmation
    volume_1h = df_1h['volume'].values
    vol_avg_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 40 to ensure sufficient data
    for i in range(40, n):
        # Skip if any required data is invalid
        if (np.isnan(cci_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current hourly volume (aligned)
        vol_1h_current = align_htf_to_ltf(prices, df_1h, volume_1h)[i]
        vol_confirm = vol_1h_current > vol_avg_20_aligned[i]
        
        # CCI extremes for mean reversion
        cci_extreme_long = cci_aligned[i] < -100
        cci_extreme_short = cci_aligned[i] > 100
        
        # Volatility filter: only trade when ATR is above its 50-period average (avoid low vol chop)
        atr_avg_50 = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean()
        atr_avg_50_val = atr_avg_50.iloc[i] if hasattr(atr_avg_50, 'iloc') else atr_avg_50[i] if i < len(atr_avg_50) else np.nan
        vol_filter = not np.isnan(atr_avg_50_val) and atr_1d_aligned[i] > atr_avg_50_val
        
        price = close[i]
        
        # Entry conditions: CCI extreme with volume and volatility confirmation
        long_signal = vol_confirm and vol_filter and cci_extreme_long
        short_signal = vol_confirm and vol_filter and cci_extreme_short
        
        # Exit conditions: CCI returns to neutral zone
        cci_neutral = (cci_aligned[i] >= -50) and (cci_aligned[i] <= 50)
        long_exit = cci_neutral
        short_exit = cci_neutral
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals