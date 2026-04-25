#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_RegimeFilter
Hypothesis: On 4h timeframe, Donchian(20) breakouts with 12h EMA50 trend filter, volume spike (>2.0x 20-bar avg), and choppiness regime filter (CHOP < 61.8 for trending) captures institutional breakouts with controlled trade frequency. Donchian channels provide objective breakout levels, 12h trend ensures alignment with intermediate-term momentum, volume spike confirms participation, and chop filter avoids whipsaws in ranging markets. Designed for 20-50 trades/year to minimize fee drag. Works in bull markets via long breakouts and bear markets via short breakouts. Uses discrete position sizing (0.25) to reduce churn. Primary timeframe: 4h, HTF: 12h.
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
    
    # Get 12h data for HTF trend and chop calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Choppiness Index on 12h (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log10(n))) / log10(n)
    # Simplified: CHOP = 100 * log10(sum(tr_range) / (ATR_max * n)) / log10(n)
    # We'll use a practical approximation: CHOP = 100 * log10(sum(tr_range) / (max(high-low) * n)) / log10(n)
    # For regime filter: CHOP < 61.8 = trending, CHOP > 38.2 = ranging
    tr_range_12h = np.maximum(high_12h - low_12h, 
                             np.maximum(np.abs(high_12h - np.concatenate([[np.nan], close_12h[:-1]])), 
                                       np.abs(low_12h - np.concatenate([[np.nan], close_12h[:-1]]))))
    tr_range_12h[0] = high_12h[0] - low_12h[0]  # first bar
    atr_sum_12h = pd.Series(tr_range_12h).rolling(window=14, min_periods=14).sum().values
    max_range_12h = pd.Series(tr_range_12h).rolling(window=14, min_periods=14).max().values
    chop_12h = 100 * np.log10(atr_sum_12h / (max_range_12h * 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate Donchian(20) channels on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 14)  # EMA50, Donchian, CHOP
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_50_aligned[i]
        chop_val = chop_aligned[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        trending_regime = chop_val < 61.8
        
        if position == 0:
            # Look for entry signals: Donchian breakout with trend, volume, and regime
            # Long: price breaks above Donchian high with uptrend (close > EMA50) and volume spike
            long_signal = (high_val > donch_high) and (close_val > ema_val) and volume_spike and trending_regime
            # Short: price breaks below Donchian low with downtrend (close < EMA50) and volume spike
            short_signal = (low_val < donch_low) and (close_val < ema_val) and volume_spike and trending_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below Donchian low (exit long)
            if close_val < donch_low:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close below EMA50 (optional early exit)
            elif close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above Donchian high (exit short)
            if close_val > donch_high:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close above EMA50 (optional early exit)
            elif close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0