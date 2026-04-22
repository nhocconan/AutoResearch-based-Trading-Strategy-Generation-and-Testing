#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 6h Donchian channels (20-period)
    price_array = prices['close'].values
    high_array = prices['high'].values
    low_array = prices['low'].values
    donch_high = pd.Series(high_array).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_array).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation
    volume_array = prices['volume'].values
    vol_ma_20 = pd.Series(volume_array).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        atr_daily = atr_14_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        price = price_array[i]
        volume = volume_array[i]
        vol_ma = vol_ma_20[i]
        
        # Volatility filter: daily ATR > 0.4 * 20-period average (avoid low volatility chop)
        atr_ma_20 = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = atr_daily > 0.4 * atr_ma_20
        
        # Volume filter: current volume > 1.2 * 20-period average
        vol_confirm = volume > 1.2 * vol_ma
        
        # Trend filter: price above/below 1d EMA50
        uptrend = price > ema50_1d_val
        downtrend = price < ema50_1d_val
        
        if position == 0:
            # Long: price breaks above 6h Donchian high + 1d uptrend + vol filter + vol confirmation
            if price > donch_high_val and uptrend and vol_filter and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h Donchian low + 1d downtrend + vol filter + vol confirmation
            elif price < donch_low_val and downtrend and vol_filter and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through opposite Donchian level or volatility drops or volume weakens
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below Donchian low or volatility collapse or weak volume
                if price < donch_low_val or not vol_filter or not vol_confirm:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above Donchian high or volatility collapse or weak volume
                if price > donch_high_val or not vol_filter or not vol_confirm:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_1dEMA50_ATRVolFilter_VolConfirm"
timeframe = "6h"
leverage = 1.0