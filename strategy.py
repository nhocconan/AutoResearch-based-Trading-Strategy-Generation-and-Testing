#!/usr/bin/env python3
"""
4h Volume Spike + ATR Regime + 1d EMA34 Trend
Hypothesis: In ranging markets (ATR ratio < 0.8), volume spikes trigger mean-reversion trades
at extremes; in trending markets (ATR ratio >= 0.8), volume spikes trigger continuation trades.
Uses 1d EMA34 for trend filter and ATR regime filter to adapt to bull/bear conditions.
Target: 30-60 trades/year on 4h timeframe.
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
    
    # Get 1d data for EMA34 trend and ATR regime (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) on 1d for regime filter
    tr1 = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1)) - np.minimum(df_1d['low'].values, np.roll(df_1d['close'].values, 1))
    tr1[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # first bar
    atr_14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate ATR(50) on 1d for long-term volatility normalization
    atr_50_1d = pd.Series(tr1).rolling(window=50, min_periods=50).mean().values
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Calculate ATR ratio (short-term/long-term) for regime detection
    atr_ratio = np.where(atr_50_1d_aligned > 0, atr_14_1d_aligned / atr_50_1d_aligned, 1.0)
    
    # Calculate volume spike: current volume > 2.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    # Calculate price position relative to 1d EMA34
    price_vs_ema = close - ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, ATR, and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_ratio[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vs_ema = price_vs_ema[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        ratio = atr_ratio[i]
        
        if position == 0:
            # Look for entry signals based on regime
            # Regime filter: ATR ratio < 0.8 = ranging (mean reversion), >= 0.8 = trending (continuation)
            if ratio < 0.8:
                # Ranging market: mean reversion at extremes
                # Long: price below EMA AND volume spike
                long_entry = (curr_vs_ema < 0) and vol_spike
                # Short: price above EMA AND volume spike
                short_entry = (curr_vs_ema > 0) and vol_spike
            else:
                # Trending market: continuation with volume
                # Long: price above EMA AND volume spike
                long_entry = (curr_vs_ema > 0) and vol_spike
                # Short: price below EMA AND volume spike
                short_entry = (curr_vs_ema < 0) and vol_spike
            
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
            # Exit: price crosses back above EMA (mean reversion) OR volume spike in opposite direction
            if (curr_vs_ema > 0) or (vol_spike and curr_vs_ema < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses back below EMA (mean reversion) OR volume spike in opposite direction
            if (curr_vs_ema < 0) or (vol_spike and curr_vs_ema > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VolumeSpike_ATRRegime_1dEMA34_Trend"
timeframe = "4h"
leverage = 1.0