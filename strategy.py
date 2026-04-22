#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume spike confirmation
# Works in bull markets via breakout momentum and in bear markets via short breakdowns
# Volume spike filters false breakouts, trend filter avoids counter-trend whipsaws
# Targets 20-50 trades/year to minimize fee drag while capturing strong moves

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period high/low) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-period rolling max/min for Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema50 = ema_50_1d_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume spike + above daily EMA50
            if price > upper and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volume spike + below daily EMA50
            elif price < lower and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on retracement to Donchian low (mean reversion) or breakdown below daily EMA50
                if price < lower or price < ema50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on retracement to Donchian high (mean reversion) or breakout above daily EMA50
                if price > upper or price > ema50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0