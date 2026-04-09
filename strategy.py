#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with weekly ATR regime filter and volume confirmation
# Uses Donchian(20) breakout from 1d for entry triggers, confirmed by volume spike (>1.5x avg)
# Only takes breakouts when weekly ATR(14) is below its 20-period MA (low volatility regime) for reliability
# Position size 0.25 to manage drawdown and enable multiple concurrent positions
# Target: 30-100 total trades over 4 years (7-25/year) to balance edge and fee drag
# Works in both bull/bear: weekly ATR regime filter ensures we trade breakouts only in low volatility environments where they are more reliable

name = "1d_1w_donchian_volume_atr_v2"
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
    
    # Load 1d data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for ATR regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = np.full(len(df_1d), np.nan)
    donchian_low = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            # Use lookback window excluding current bar to avoid look-ahead
            donchian_high[i] = np.max(df_1d['high'].iloc[i-20:i])
            donchian_low[i] = np.min(df_1d['low'].iloc[i-20:i])
    
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
    
    # Calculate 20-period MA of ATR for regime filter
    atr_ma_20 = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        if i < 20:
            atr_ma_20[i] = np.nan
        else:
            atr_ma_20[i] = np.mean(atr_1w[i-20:i])
    
    # Align 1d Donchian channels to 1d timeframe (no shift needed as already aligned)
    donchian_high_1d = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_1d = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Align weekly ATR regime to 1d timeframe
    atr_ma_20_1d = align_htf_to_ltf(prices, df_1w, atr_ma_20)
    atr_1w_1d = align_htf_to_ltf(prices, df_1w, atr_1w)
    
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
        if (np.isnan(donchian_high_1d[i]) or 
            np.isnan(donchian_low_1d[i]) or 
            np.isnan(atr_ma_20_1d[i]) or 
            np.isnan(atr_1w_1d[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        # ATR regime filter: only trade when current ATR < ATR MA (low volatility regime)
        atr_regime = atr_1w_1d[i] < atr_ma_20_1d[i]
        
        if position == 1:  # Long position
            # Exit conditions: price closes below Donchian low OR ATR regime turns unfavorable
            if close[i] < donchian_low_1d[i] or not atr_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Donchian high OR ATR regime turns unfavorable
            if close[i] > donchian_high_1d[i] or not atr_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout with volume confirmation and ATR regime filter
            if volume_confirm and atr_regime:
                # Long breakout: price closes above Donchian high
                if close[i] > donchian_high_1d[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian low
                elif close[i] < donchian_low_1d[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals