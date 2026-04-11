#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot + 1w volume confirmation + choppiness regime filter
# - Long when price touches Camarilla L3 support with volume > 1.5x 1w average and chop > 61.8 (range)
# - Short when price touches Camarilla H3 resistance with volume > 1.5x 1w average and chop > 61.8 (range)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work in ranging markets (chop > 61.8) which are common in bear/consolidation periods
# - 1w HTF provides reliable volume confirmation, 1d timeframe balances signal quality and cost

name = "1d_1w_camarilla_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data for Camarilla pivots and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Load 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1d Camarilla pivots (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Pre-compute 1d choppiness index (14-period)
    # True range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Choppiness = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum()
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop_1d = 100 * np.log10(atr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_1d = chop_1d.values
    
    # Pre-compute 1w volume SMA (20-period)
    volume_1w = df_1w['volume'].values
    volume_series = pd.Series(volume_1w)
    volume_sma_20_1w = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 1d timeframe (no alignment needed for same timeframe)
    camarilla_h3_aligned = camarilla_h3
    camarilla_l3_aligned = camarilla_l3
    chop_1d_aligned = chop_1d
    
    # Align 1w volume to 1d timeframe
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla touch conditions (using today's price vs yesterday's levels)
        touch_h3 = abs(price_high - camarilla_h3_aligned[i-1]) < 0.001 * camarilla_h3_aligned[i-1]  # Within 0.1%
        touch_l3 = abs(price_low - camarilla_l3_aligned[i-1]) < 0.001 * camarilla_l3_aligned[i-1]  # Within 0.1%
        
        # Volume confirmation: current volume > 1.5x 20-period average (using 1w aligned volume)
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Choppiness regime: chop > 61.8 indicates ranging market (good for mean reversion at pivots)
        chop_regime = chop_1d_aligned[i] > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla L3 touch + volume confirmation + chop regime
        if touch_l3 and vol_confirm and chop_regime:
            enter_long = True
        
        # Short: Camarilla H3 touch + volume confirmation + chop regime
        if touch_h3 and vol_confirm and chop_regime:
            enter_short = True
        
        # Exit conditions: opposite Camarilla touch or chop regime breakdown
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price touches H3 OR chop regime breaks down
            exit_long = touch_h3 or (not chop_regime)
        elif position == -1:
            # Exit short if price touches L3 OR chop regime breaks down
            exit_short = touch_l3 or (not chop_regime)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals