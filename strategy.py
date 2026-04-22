#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    # Load 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
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
    
    # Align daily ATR to 1h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 1h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Price array
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        atr_daily = atr_14_aligned[i]
        ema50_4h_val = ema50_4h_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volatility filter: daily ATR > 0.5 * 20-period average (avoid low volatility chop)
        atr_ma_20 = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = atr_daily > 0.5 * atr_ma_20
        
        # Volume filter: current volume > 1.5 * 20-period average volume
        vol_spike = vol > 1.5 * vol_ma
        
        # Trend filter: price above/below 4h EMA50
        uptrend = price > ema50_4h_val
        downtrend = price < ema50_4h_val
        
        if position == 0:
            # Long: price breaks above 1h Donchian high + 4h uptrend + volatility filter + volume spike
            if price > donch_high_val and uptrend and vol_filter and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 1h Donchian low + 4h downtrend + volatility filter + volume spike
            elif price < donch_low_val and downtrend and vol_filter and vol_spike:
                signals[i] = -0.20
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
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Donchian20_4hEMA50_ATRVolFilter_VolSpike_Session"
timeframe = "1h"
leverage = 1.0