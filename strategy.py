#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter.
# Long when price breaks above Donchian upper band AND 1d volume > 1.5x 20-period average AND chop < 61.8 (trending regime).
# Short when price breaks below Donchian lower band AND 1d volume > 1.5x 20-period average AND chop < 61.8.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 20-40 trades/year.
# Donchian channels provide clear breakout levels, volume confirms conviction, chop filter avoids whipsaws in ranging markets.
# Works in bull (breakouts continuation) and bear (breakdowns continuation) by following institutional price action.

name = "4h_Donchian20_1dVolumeConfirm_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume SMA(20)
    vol_1d = df_1d['volume'].values
    vol_sma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Chopiness Index(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index = 100 * log10(sum(ATR(14)) / (HHV(14) - LLV(14))) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_14 - ll_14
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((range_14 == 0) | np.isnan(range_14) | np.isnan(sum_atr_14), 50.0, chop)
    
    # Align 1d indicators to 4h
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian(20) on 4h data
    # Upper band = 20-period high
    # Lower band = 20-period low
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for Donchian and 1d indicators
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_sma_20_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_donch_high = donch_high[i]
        curr_donch_low = donch_low[i]
        curr_vol_sma_20 = vol_sma_20_aligned[i]
        curr_chop = chop_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = curr_vol > 1.5 * curr_vol_sma_20
        
        # Chop regime filter: chop < 61.8 indicates trending regime (avoid ranging markets)
        trending_regime = curr_chop < 61.8
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band AND volume confirm AND trending regime
            if (curr_close > curr_donch_high and 
                vol_confirm and 
                trending_regime):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND volume confirm AND trending regime
            elif (curr_close < curr_donch_low and 
                  vol_confirm and 
                  trending_regime):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower band OR chop > 61.8 (ranging) OR volume drops below average
            if (curr_close < curr_donch_low or 
                curr_chop > 61.8 or 
                curr_vol < curr_vol_sma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band OR chop > 61.8 (ranging) OR volume drops below average
            if (curr_close > curr_donch_high or 
                curr_chop > 61.8 or 
                curr_vol < curr_vol_sma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals