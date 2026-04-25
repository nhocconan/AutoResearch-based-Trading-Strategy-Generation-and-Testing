#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout with Volume Spike and ATR Regime Filter
Hypothesis: Camarilla pivot levels (H3/L3) act as strong support/resistance. 
Breakouts above H3 or below L3 with volume confirmation and ATR-based volatility regime filter 
capture institutional moves. Uses ATR percentile to distinguish trending vs ranging markets.
Works in both bull/bear markets: in bull, longs above H3 with high volatility regime; 
in bear, shorts below L3 with high volatility regime. Discrete sizing (0.25) limits fee drag.
Target: 12-30 trades/year (50-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d ATR for volatility regime filter (using 14 periods)
    tr1 = df_1d['high'][1:] - df_1d['low'][1:]
    tr2 = np.abs(df_1d['high'][1:] - df_1d['close'][:-1])
    tr3 = np.abs(df_1d['low'][1:] - df_1d['close'][:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate ATR percentile rank (50-period lookback) for regime filter
    atr_percentile = pd.Series(atr_14_1d).rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate Camarilla pivot levels (H3, L3) from prior 1d bar
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 4
    camarilla_l3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for ATR percentile calculation
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(atr_percentile_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        atr_regime = atr_percentile_aligned[i]  # ATR percentile rank (0-100)
        camarilla_h3_val = camarilla_h3_aligned[i]
        camarilla_l3_val = camarilla_l3_aligned[i]
        
        # Volume spike: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 1.5 * vol_ma_20
        
        # Regime filter: only trade in high volatility regimes (ATR percentile > 60)
        high_vol_regime = atr_regime > 60
        
        # Breakout conditions: price breaks above H3 or below L3
        bullish_breakout = curr_close > camarilla_h3_val
        bearish_breakout = curr_close < camarilla_l3_val
        
        # Exit conditions: reverse breakout or low volatility regime
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish breakout below L3 or low volatility regime
                if bearish_breakout or not high_vol_regime:
                    exit_signal = True
                    
            elif position == -1:
                # Exit on bullish breakout above H3 or low volatility regime
                if bullish_breakout or not high_vol_regime:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Camarilla breakout + volume spike + high volatility regime
        if position == 0:
            # Long: break above H3 AND volume spike AND high vol regime
            long_condition = bullish_breakout and volume_spike and high_vol_regime
            # Short: break below L3 AND volume spike AND high vol regime
            short_condition = bearish_breakout and volume_spike and high_vol_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_VolumeSpike_ATRRegime_v1"
timeframe = "12h"
leverage = 1.0