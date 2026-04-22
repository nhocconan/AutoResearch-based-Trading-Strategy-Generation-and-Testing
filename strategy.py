#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index (14) + 1d RSI (14) reversal with volume confirmation.
# Choppiness Index > 61.8 indicates ranging market (mean reversion opportunity).
# RSI < 30 or > 70 indicates extreme conditions. Combined with volume spike (>1.5x 20-period avg),
# this captures mean reversion in ranging markets while avoiding strong trends.
# Works in both bull and bear markets by fading extremes in ranging conditions.
# Designed for low trade frequency (~20-30/year) to minimize fee decay.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Choppiness Index calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) for Choppiness Index
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original length
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Calculate Choppiness Index: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr14 / (hh_14 - ll_14)) / np.log10(14)
    
    # Calculate 14-period RSI on 1d close
    delta = pd.Series(close_1d).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, np.inf, rs)  # avoid division by zero
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to 4h timeframe (waits for 1d bar to close)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop_aligned[i]
        rsi_val = rsi_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        # Choppiness > 61.8 indicates ranging market
        ranging = chop_val > 61.8
        
        if position == 0:
            # Long conditions: ranging market + RSI oversold + volume spike
            if ranging and rsi_val < 30 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: ranging market + RSI overbought + volume spike
            elif ranging and rsi_val > 70 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI returns to neutral (50) or chop drops below 50 (trending)
                if rsi_val >= 50 or chop_val < 50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI returns to neutral (50) or chop drops below 50 (trending)
                if rsi_val <= 50 or chop_val < 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Choppiness_RSI_Reversal_Volume"
timeframe = "4h"
leverage = 1.0