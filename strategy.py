#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for 1d EMA34 and ATR14
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d ATR14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Load 1w data for weekly trend
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Price and volume arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        ema34_1d_val = ema34_1d_aligned[i]
        ema34_1w_val = ema34_1w_aligned[i]
        atr_daily = atr14_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volatility filter: daily ATR > 0.3 * 20-period average ATR (avoid low volatility chop)
        atr_ma_20 = pd.Series(atr14_1d_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = atr_daily > 0.3 * atr_ma_20
        
        # Volume filter: current volume > 1.3 * 20-period average volume
        vol_spike = vol > 1.3 * vol_ma
        
        # Trend filter: price above/below 1d EMA34 AND weekly trend alignment
        uptrend = price > ema34_1d_val and ema34_1d_val > ema34_1w_val
        downtrend = price < ema34_1d_val and ema34_1d_val < ema34_1w_val
        
        if position == 0:
            # Long: price breaks above 4h Donchian high + uptrend + volatility filter + volume spike
            if price > donch_high_val and uptrend and vol_filter and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian low + downtrend + volatility filter + volume spike
            elif price < donch_low_val and downtrend and vol_filter and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through opposite Donchian level or volatility drops or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below Donchian low or volatility collapse or volume drop
                if price < donch_low_val or not vol_filter or not vol_spike:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above Donchian high or volatility collapse or volume drop
                if price > donch_high_val or not vol_filter or not vol_spike:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_1wTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0