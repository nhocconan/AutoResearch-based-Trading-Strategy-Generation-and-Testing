#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 2.0x 20-period volume SMA AND chop < 61.8 (trending)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 2.0x 20-period volume SMA AND chop < 61.8 (trending)
# - Exit: price returns to Camarilla Pivot level (midpoint) or opposite H3/L3 break
# - Position sizing: 0.25 discrete level to minimize fee churn and manage drawdown
# - Uses Camarilla pivots from 1d for structure, volume for confirmation, chop regime to avoid whipsaws
# - Target: 20-40 trades/year on 4h timeframe to stay within fee drag limits

name = "4h_1d_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Camarilla: H4 = close + 1.1*(high-low)/2, H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4, L4 = close - 1.1*(high-low)/2
    # Pivot = (high + low + close)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4.0
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4.0
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_4h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate Choppiness Index regime filter (14-period)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_period = 14
    chop_period = 14
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Sum of ATR over chop_period
    atr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    # Max high - min low over chop_period
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    range_chop = max_high - min_low
    
    # Choppiness Index
    chop = np.zeros(n)
    for i in range(chop_period, n):
        if atr_sum[i] > 0 and range_chop[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_chop[i]) / np.log10(chop_period)
        else:
            chop[i] = 50.0  # neutral
    
    for i in range(chop_period, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_4h[i]) or np.isnan(camarilla_l3_4h[i]) or
            np.isnan(camarilla_pivot_4h[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume SMA
        # Each 1d bar = 6 4h bars
        idx_1d = i // 6
        if idx_1d < len(volume_1d):
            vol_confirm = volume_1d[idx_1d] > 2.0 * volume_sma_20_1d_aligned[i]
        else:
            vol_confirm = False
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        trending_regime = chop[i] < 61.8
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3_4h[i]  # Break above H3
        breakout_down = close[i] < camarilla_l3_4h[i]  # Break below L3
        
        # Exit conditions: return to pivot or opposite break
        exit_long = close[i] < camarilla_pivot_4h[i]
        exit_short = close[i] > camarilla_pivot_4h[i]
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm and trending_regime:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm and trending_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long or breakout_down:  # Exit on pivot return or opposite break
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short or breakout_up:  # Exit on pivot return or opposite break
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals