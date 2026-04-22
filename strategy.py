#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# Long when price breaks above upper Donchian + ATR(1d) > 20-period MA + volume spike
# Short when price breaks below lower Donchian + ATR(1d) > 20-period MA + volume spike
# Exit when price returns to opposite Donchian band
# Designed for low trade frequency (~10-25/year) with strong trend-following edge in volatile markets
# ATR filter ensures trades only in high-volatility regimes, reducing whipsaws in ranging markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ATR calculation and Donchian bands
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period ATR on daily timeframe
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period high/low) on daily timeframe
    upper_donch = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_donch = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align ATR and Donchian levels to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    upper_donch_aligned = align_htf_to_ltf(prices, df_1d, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_1d, lower_donch)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(upper_donch_aligned[i]) or 
            np.isnan(lower_donch_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr_aligned[i]
        upper_donch_val = upper_donch_aligned[i]
        lower_donch_val = lower_donch_aligned[i]
        
        # Volatility filter: current ATR > 20-period ATR average (high volatility regime)
        vol_regime = atr_val > np.nanmean(atr_20) if not np.isnan(np.nanmean(atr_20)) else False
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + high volatility + volume spike
            if price > upper_donch_val and vol_regime and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + high volatility + volume spike
            elif price < lower_donch_val and vol_regime and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to opposite Donchian band
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to or below lower Donchian band
                if price <= lower_donch_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to or above upper Donchian band
                if price >= upper_donch_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dATR_Volume"
timeframe = "12h"
leverage = 1.0