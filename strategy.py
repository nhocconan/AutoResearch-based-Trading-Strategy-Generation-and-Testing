#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour ATR breakout with 1-day trend filter and volume confirmation
# Uses daily ATR to set dynamic breakout levels from the prior day's close
# Daily ADX > 25 filters for trending conditions to avoid false breakouts in ranging markets
# Volume > 1.5x 6-period average confirms institutional participation
# Designed for 12-30 trades per year (~50-120 total over 4 years) to minimize fee drag
# Works in bull markets via upward breakouts and in bear markets via downward breakdowns

name = "12h_1d_atr_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period)
    atr_1d = np.full(len(df_1d), np.nan)
    tr_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        tr0 = df_1d['high'].iloc[i] - df_1d['low'].iloc[i]
        tr1 = abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1])
        tr2 = abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        tr_1d[i] = max(tr0, tr1, tr2)
    
    if len(df_1d) >= 14:
        atr_1d[13] = np.nansum(tr_1d[1:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate daily ADX (14-period)
    adx_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 28:
        # Calculate +DM and -DM
        dm_plus = np.zeros(len(df_1d))
        dm_minus = np.zeros(len(df_1d))
        for i in range(1, len(df_1d)):
            up_move = df_1d['high'].iloc[i] - df_1d['high'].iloc[i-1]
            down_move = df_1d['low'].iloc[i-1] - df_1d['low'].iloc[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smooth TR, +DM, -DM
        tr14 = np.full(len(df_1d), np.nan)
        dm_plus_14 = np.full(len(df_1d), np.nan)
        dm_minus_14 = np.full(len(df_1d), np.nan)
        tr14[13] = np.nansum(tr_1d[1:14])
        dm_plus_14[13] = np.nansum(dm_plus[1:14])
        dm_minus_14[13] = np.nansum(dm_minus[1:14])
        for i in range(14, len(df_1d)):
            tr14[i] = tr14[i-1] - (tr14[i-1] / 14) + tr_1d[i]
            dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
            dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
        
        # Calculate DI and DX
        di_plus = np.full(len(df_1d), np.nan)
        di_minus = np.full(len(df_1d), np.nan)
        dx = np.full(len(df_1d), np.nan)
        for i in range(14, len(df_1d)):
            if tr14[i] > 0:
                di_plus[i] = 100 * dm_plus_14[i] / tr14[i]
                di_minus[i] = 100 * dm_minus_14[i] / tr14[i]
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        
        # Calculate ADX (smoothed DX)
        if len(df_1d) >= 28:
            adx_1d[27] = np.nansum(dx[14:28]) / 14
            for i in range(28, len(df_1d)):
                adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    # Align 1d values to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 6-period average (3d)
    vol_ma_6 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 6:
            vol_sum -= volume[i-6]
        if i >= 5:
            vol_ma_6[i] = vol_sum / 6
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(28, n):  # Start after ADX warmup
        # Skip if any required data is invalid
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_6[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below prior day's close OR ADX drops below 20
            if (close[i] <= df_1d['close'].iloc[-1] if len(df_1d) > 0 else 0) or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above prior day's close OR ADX drops below 20
            if (close[i] >= df_1d['close'].iloc[-1] if len(df_1d) > 0 else 0) or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate breakout levels using prior day's close and ATR
            if len(df_1d) >= 2:
                prev_close = df_1d['close'].iloc[-2]
                atr_val = atr_aligned[i]
                
                # Enter long: price breaks above prior close + 0.5*ATR with volume confirmation AND ADX > 25
                vol_ratio = volume[i] / vol_ma_6[i] if vol_ma_6[i] > 0 else 0
                if (close[i] > prev_close + 0.5 * atr_val and 
                    vol_ratio > 1.5 and 
                    adx_aligned[i] > 25):
                    position = 1
                    signals[i] = 0.25
                # Enter short: price breaks below prior close - 0.5*ATR with volume confirmation AND ADX > 25
                elif (close[i] < prev_close - 0.5 * atr_val and 
                      vol_ratio > 1.5 and 
                      adx_aligned[i] > 25):
                    position = -1
                    signals[i] = -0.25
    
    return signals