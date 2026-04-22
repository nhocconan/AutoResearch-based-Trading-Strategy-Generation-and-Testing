#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    # Load 1w data for volatility regime
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1w ATR for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ATR to 4h timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate 4-hour Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Price array
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4-hour volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        atr_weekly = atr_1w_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        # Volatility filter: weekly ATR > 50th percentile of its 50-period range (avoid low volatility chop)
        atr_min = pd.Series(atr_1w_aligned).rolling(window=50, min_periods=50).min().values[i]
        atr_max = pd.Series(atr_1w_aligned).rolling(window=50, min_periods=50).max().values[i]
        atr_percentile = (atr_weekly - atr_min) / (atr_max - atr_min + 1e-10)
        vol_filter = atr_percentile > 0.5
        
        # Trend filter: price above/below 1d EMA50
        uptrend = price > ema50_1d_val
        downtrend = price < ema50_1d_val
        
        # Volume filter: volume above average
        vol_filter = vol > vol_ma_val
        
        if position == 0:
            # Long: price breaks above 4h Donchian high + 1d uptrend + volatility filter + volume filter
            if price > donch_high_val and uptrend and vol_filter and vol_filter_vol:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian low + 1d downtrend + volatility filter + volume filter
            elif price < donch_low_val and downtrend and vol_filter and vol_filter_vol:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through opposite Donchian level or volatility drops or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below Donchian low or volatility collapse or volume drop
                if price < donch_low_val or not vol_filter or not vol_filter_vol:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above Donchian high or volatility collapse or volume drop
                if price > donch_high_val or not vol_filter or not vol_filter_vol:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_1wATRVol_VolFilter"
timeframe = "4h"
leverage = 1.0