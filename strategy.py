#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout + 1d ATR Regime + Volume Spike
Hypothesis: Camarilla H3/L3 levels represent stronger support/resistance than R3/S3. Breakout above H3 or below L3 with volume confirmation in trending markets (ATR regime filter) captures sustained moves. Uses ATR(14) to filter choppy regimes - only trade when ATR(14) > ATR(50) indicating trending conditions. Works in bull (long on H3 break) and bear (short on L3 break). Target: 20-40 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots, EMA34 trend, and ATR regime (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ATR(14) and ATR(50) for regime filter on 1d
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = tr.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR regime: trending when ATR(14) > ATR(50)
    atr_regime = atr_14 > atr_50
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # Calculate EMA34 on 1d for trend filter
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels on 1d (H3, L3)
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    high_1d_arr = df_1d['high'].values
    low_1d_arr = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    camarilla_h3 = close_1d_arr + 1.1 * (high_1d_arr - low_1d_arr) / 4
    camarilla_l3 = close_1d_arr - 1.1 * (high_1d_arr - low_1d_arr) / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for ATR(50), EMA34, volume MA
    start_idx = max(50, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_regime_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        is_trending = atr_regime_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price > H3, above 1d EMA34, volume confirmation, trending regime
            long_entry = (curr_close > h3_level) and (curr_close > ema_34_val) and volume_confirm and is_trending
            # Short: price < L3, below 1d EMA34, volume confirmation, trending regime
            short_entry = (curr_close < l3_level) and (curr_close < ema_34_val) and volume_confirm and is_trending
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below 1d EMA34 OR price breaks below L3 (stop and reverse)
            if curr_close < ema_34_val or curr_close < l3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above 1d EMA34 OR price breaks above H3 (stop and reverse)
            if curr_close > ema_34_val or curr_close > h3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dATR_Regime_VolumeSpike"
timeframe = "4h"
leverage = 1.0