#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1d ATR-based breakout with volume confirmation
# Long when price breaks above ATR(14) upper band AND Choppiness < 38.2 (trending) AND volume spike
# Short when price breaks below ATR(14) lower band AND Choppiness < 38.2 (trending) AND volume spike
# Exit when price returns to ATR midline OR Choppiness > 61.8 (ranging) OR volatility drops
# Designed for low trade frequency with regime awareness - avoids whipsaws in ranging markets
# Uses ATR-based channels for volatility-adjusted breakouts and Choppiness Index for regime detection

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for ATR calculation and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) for volatility-based channels
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR midline (average of high and low)
    atr_mid = (high_1d + low_1d) / 2
    atr_upper = atr_mid + atr * 1.5  # Upper band
    atr_lower = atr_mid - atr * 1.5  # Lower band
    
    # Align ATR bands to 12h timeframe
    atr_upper_aligned = align_htf_to_ltf(prices, df_1d, atr_upper)
    atr_lower_aligned = align_htf_to_ltf(prices, df_1d, atr_lower)
    atr_mid_aligned = align_htf_to_ltf(prices, df_1d, atr_mid)
    
    # Calculate Choppiness Index for regime detection
    # CHOP = 100 * log10(sum(TR over n) / (HH(n) - LL(n))) / log10(n)
    period = 14
    sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high_1d).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low_1d).rolling(window=period, min_periods=period).min().values
    chop_raw = 100 * np.log10(sum_tr / (hh - ll + 1e-10)) / np.log10(period)
    chop_raw[hh == ll] = 50  # Avoid division by zero
    chop = np.where((hh - ll) > 0, chop_raw, 50)
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_upper_aligned[i]) or 
            np.isnan(atr_lower_aligned[i]) or 
            np.isnan(atr_mid_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_upper_val = atr_upper_aligned[i]
        atr_lower_val = atr_lower_aligned[i]
        atr_mid_val = atr_mid_aligned[i]
        chop_val = chop_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        # Regime filters: trending market (CHOP < 38.2) vs ranging (CHOP > 61.8)
        trending = chop_val < 38.2
        ranging = chop_val > 61.8
        
        if position == 0:
            # Long conditions: price breaks above ATR upper band AND trending AND volume spike
            if price > atr_upper_val and trending and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below ATR lower band AND trending AND volume spike
            elif price < atr_lower_val and trending and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to ATR midline OR market becomes ranging OR volatility drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to ATR midline or market ranges
                if price <= atr_mid_val or ranging:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to ATR midline or market ranges
                if price >= atr_mid_val or ranging:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_ATR_Choppiness_Breakout_Volume"
timeframe = "12h"
leverage = 1.0