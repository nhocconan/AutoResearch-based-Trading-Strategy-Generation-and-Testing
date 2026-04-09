#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ATR regime filter
# Uses 12h price breaking above/below Camarilla pivot levels (H3/L3) derived from 1d OHLC
# Only takes breakouts when 1d volume > 1.5x 20-period average AND 1d ATR < ATR MA (low volatility)
# Position size 0.25 to manage drawdown and enable multiple concurrent positions
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in both bull/bear: 1d ATR regime filter ensures breakouts occur in low volatility environments where they are more reliable

name = "12h_1d_camarilla_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volume and ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.full(len(df_1d), np.nan)
    atr_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        tr = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
        tr_1d[i] = tr
    
    # Calculate ATR with Wilder's smoothing
    for i in range(len(df_1d)):
        if i < 14:
            atr_1d[i] = np.nan
        elif i == 14:
            atr_1d[i] = np.nanmean(tr_1d[1:15])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate 50-period MA of ATR for regime filter
    atr_ma_50 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i < 50:
            atr_ma_50[i] = np.nan
        else:
            atr_ma_50[i] = np.mean(atr_1d[i-50:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume_1d = np.full(len(df_1d), np.nan)
    vol_1d = df_1d['volume'].values
    for i in range(len(df_1d)):
        if i < 20:
            avg_volume_1d[i] = np.nan
        else:
            avg_volume_1d[i] = np.mean(vol_1d[i-20:i])
    
    # Align 1d indicators to 12h timeframe
    atr_ma_50_12h = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    avg_volume_12h = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate 12h Camarilla pivot levels from 1d OHLC
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # We use H3 and L3 as breakout levels
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    for i in range(n):
        # Need 1d data that is complete (shifted by 1 to avoid look-ahead)
        idx_1d = i // 2  # Approximate: 2x 12h bars per 1d
        if idx_1d < 1 or idx_1d >= len(df_1d):
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            continue
            
        # Use previous 1d OHLC to avoid look-ahead
        prev_high = high_1d[idx_1d-1]
        prev_low = low_1d[idx_1d-1]
        prev_close = close_1d[idx_1d-1]
        
        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_ = prev_high - prev_low
        
        # Camarilla levels
        camarilla_h3[i] = pivot + range_ * 1.1 / 4.0
        camarilla_l3[i] = pivot - range_ * 1.1 / 4.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or 
            np.isnan(atr_ma_50_12h[i]) or 
            np.isnan(atr_12h[i]) or 
            np.isnan(avg_volume_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirm = df_1d['volume'].iloc[min(i//2, len(df_1d)-1)] > 1.5 * avg_volume_12h[i] if i//2 < len(df_1d) else False
        
        # ATR regime filter: only trade when current ATR < ATR MA (low volatility regime)
        atr_regime = atr_12h[i] < atr_ma_50_12h[i]
        
        if position == 1:  # Long position
            # Exit conditions: price closes below Camarilla L3 OR ATR regime turns unfavorable
            if close[i] < camarilla_l3[i] or not atr_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Camarilla H3 OR ATR regime turns unfavorable
            if close[i] > camarilla_h3[i] or not atr_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla breakout with volume confirmation and ATR regime filter
            if volume_confirm and atr_regime:
                # Long breakout: price closes above Camarilla H3
                if close[i] > camarilla_h3[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Camarilla L3
                elif close[i] < camarilla_l3[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals