#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot points (CPR) for trend direction and 1d ATR-based volatility regime filter
# Uses weekly Central Pivot Range (CPR: TC, BC, PP) to determine market bias (above/below CPR)
# Only takes long when price > weekly TC and short when price < weekly BC
# Volatility filter: trade only when 1d ATR(14) is below its 20-period MA (low volatility regime) for reliable breakouts
# Position size 0.25 to manage drawdown and enable multiple concurrent positions
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag
# Works in both bull/bear: volatility regime filter ensures we trade only in low volatility environments where CPR breaks are more reliable

name = "6h_1w_1d_cpr_volume_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for weekly CPR
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly CPR (TC, BC, PP) from previous week's OHLC to avoid look-ahead
    weekly_tc = np.full(len(df_1w), np.nan)  # Top Central Pivot
    weekly_bc = np.full(len(df_1w), np.nan)  # Bottom Central Pivot
    weekly_pp = np.full(len(df_1w), np.nan)  # Pivot Point
    
    for i in range(len(df_1w)):
        if i < 1:
            weekly_tc[i] = np.nan
            weekly_bc[i] = np.nan
            weekly_pp[i] = np.nan
        else:
            # Use previous week's OHLC to calculate CPR
            prev_high = df_1w['high'].iloc[i-1]
            prev_low = df_1w['low'].iloc[i-1]
            prev_close = df_1w['close'].iloc[i-1]
            
            # Pivot Point
            weekly_pp[i] = (prev_high + prev_low + prev_close) / 3.0
            
            # Central Pivot Range (CPR)
            weekly_tc[i] = (weekly_pp[i] + max(prev_high, prev_low)) / 2.0
            weekly_bc[i] = (weekly_pp[i] + min(prev_high, prev_low)) / 2.0
    
    # Load 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr_1d = np.full(len(df_1d), np.nan)
    atr_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        tr = max(
            df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
            abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
            abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
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
    
    # Calculate 20-period MA of ATR for regime filter
    atr_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i < 20:
            atr_ma_20[i] = np.nan
        else:
            atr_ma_20[i] = np.mean(atr_1d[i-20:i])
    
    # Align HTF indicators to 6h timeframe
    weekly_tc_6h = align_htf_to_ltf(prices, df_1w, weekly_tc)
    weekly_bc_6h = align_htf_to_ltf(prices, df_1w, weekly_bc)
    weekly_pp_6h = align_htf_to_ltf(prices, df_1w, weekly_pp)
    atr_ma_20_6h = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(weekly_tc_6h[i]) or 
            np.isnan(weekly_bc_6h[i]) or 
            np.isnan(weekly_pp_6h[i]) or 
            np.isnan(atr_ma_20_6h[i]) or 
            np.isnan(atr_6h[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        # Volatility regime filter: only trade when current ATR < ATR MA (low volatility regime)
        vol_regime = atr_6h[i] < atr_ma_20_6h[i]
        
        if position == 1:  # Long position
            # Exit conditions: price closes below weekly PP OR volatility regime turns unfavorable
            if close[i] < weekly_pp_6h[i] or not vol_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above weekly PP OR volatility regime turns unfavorable
            if close[i] > weekly_pp_6h[i] or not vol_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: CPR breakout with volume confirmation and volatility regime filter
            if volume_confirm and vol_regime:
                # Long breakout: price closes above weekly TC (bullish bias)
                if close[i] > weekly_tc_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below weekly BC (bearish bias)
                elif close[i] < weekly_bc_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals