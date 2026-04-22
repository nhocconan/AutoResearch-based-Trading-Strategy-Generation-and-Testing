#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + weekly Bollinger Band squeeze + volume confirmation.
# Uses weekly Bollinger Band width to detect low volatility (squeeze) conditions.
# When squeeze occurs, trade in direction of daily KAMA with volume confirmation.
# Designed to work in both bull and bear markets by capturing breakouts from low volatility.
# Targets 10-25 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for Bollinger Band squeeze (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # Bollinger Band squeeze: BB width below 20-period average
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma
    
    # Align BB squeeze to daily timeframe
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1w, bb_squeeze)
    
    # Load daily data for KAMA
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (2, 10, 30)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, 10))
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=0)
    er = np.zeros_like(close_1d)
    er[10:] = change[10:] / volatility[10:]
    er[volatility == 0] = 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (already daily, but align for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate daily ATR for stop loss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to daily timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Daily volume moving average
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bb_squeeze_aligned[i]) or 
            np.isnan(kama_aligned[i]) or 
            np.isnan(atr_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume_1d[i]  # Use daily volume for volume confirmation
        vol_ma = vol_ma_20[i]
        kama_val = kama_aligned[i]
        bb_squeeze_val = bb_squeeze_aligned[i]
        atr_val = atr_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_confirm = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter only during Bollinger Band squeeze with volume confirmation
            if bb_squeeze_val and vol_confirm:
                if price > kama_val:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif price < kama_val:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position != 0:
            # Exit conditions: stop loss or mean reversion
            exit_signal = False
            
            if position == 1:  # long position
                # Stop loss: 2 * ATR below entry
                if price < entry_price - 2.0 * atr_val:
                    exit_signal = True
                # Mean reversion: price returns to KAMA
                elif price <= kama_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Stop loss: 2 * ATR above entry
                if price > entry_price + 2.0 * atr_val:
                    exit_signal = True
                # Mean reversion: price returns to KAMA
                elif price >= kama_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_BBSqueeze_Volume"
timeframe = "1d"
leverage = 1.0