#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with weekly volume confirmation and ATR regime filter
# Uses Donchian(20) from 1d for breakout triggers, confirmed by weekly volume spike (>2.0x avg)
# Only takes breakouts when weekly ATR(14) is below its 50-period MA (low volatility regime) for reliability
# Position size 0.25 to manage drawdown and enable multiple concurrent positions
# Target: 30-100 total trades over 4 years (7-25/year) to balance edge and fee drag
# Works in both bull/bear: weekly ATR regime filter ensures we trade breakouts only in low volatility environments where they are more reliable

name = "1d_1w_donchian_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for volume confirmation and ATR regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate weekly ATR(14) for regime filter
    tr_1w = np.full(len(df_1w), np.nan)
    atr_1w = np.full(len(df_1w), np.nan)
    
    for i in range(1, len(df_1w)):
        tr = max(
            df_1w['high'].iloc[i] - df_1w['low'].iloc[i],
            abs(df_1w['high'].iloc[i] - df_1w['close'].iloc[i-1]),
            abs(df_1w['low'].iloc[i] - df_1w['close'].iloc[i-1])
        )
        tr_1w[i] = tr
    
    # Calculate ATR with Wilder's smoothing
    for i in range(len(df_1w)):
        if i < 14:
            atr_1w[i] = np.nan
        elif i == 14:
            atr_1w[i] = np.nanmean(tr_1w[1:15])
        else:
            atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    # Calculate 50-period MA of ATR for regime filter
    atr_ma_50_1w = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        if i < 50:
            atr_ma_50_1w[i] = np.nan
        else:
            atr_ma_50_1w[i] = np.mean(atr_1w[i-50:i])
    
    # Calculate weekly average volume for volume confirmation
    avg_volume_1w = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        if i < 20:
            avg_volume_1w[i] = np.nan
        else:
            avg_volume_1w[i] = np.mean(df_1w['volume'].iloc[i-20:i])
    
    # Align weekly indicators to 1d timeframe
    atr_ma_50_1d = align_htf_to_ltf(prices, df_1w, atr_ma_50_1w)
    atr_1d = align_htf_to_ltf(prices, df_1w, atr_1w)
    avg_volume_1d = align_htf_to_ltf(prices, df_1w, avg_volume_1w)
    
    # Calculate 1d Donchian channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr_ma_50_1d[i]) or 
            np.isnan(atr_1d[i]) or 
            np.isnan(avg_volume_1d[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period weekly average (aligned)
        volume_confirm = volume[i] > 2.0 * avg_volume_1d[i]
        
        # ATR regime filter: only trade when current ATR < ATR MA (low volatility regime)
        atr_regime = atr_1d[i] < atr_ma_50_1d[i]
        
        if position == 1:  # Long position
            # Exit conditions: price closes below Donchian low OR ATR regime turns unfavorable
            if close[i] < donchian_low[i] or not atr_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Donchian high OR ATR regime turns unfavorable
            if close[i] > donchian_high[i] or not atr_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout with volume confirmation and ATR regime filter
            if volume_confirm and atr_regime:
                # Long breakout: price closes above Donchian high
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian low
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals