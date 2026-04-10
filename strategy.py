#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1w ATR regime filter + volume confirmation
# - Primary: 6h price breaks above/below Donchian channel (20-period high/low)
# - HTF: 1w ATR(14) > 1.5x its 50-period EMA = high volatility regime (trend follow)
# - Volume confirmation: 6h volume > 1.5x 20-period EMA
# - Long: Price breaks above Donchian upper + ATR regime + volume confirmation
# - Short: Price breaks below Donchian lower + ATR regime + volume confirmation
# - Exit: Price returns to Donchian midpoint (mean reversion within channel)
# - Position sizing: 0.25 (discrete level, balances return/drawdown, reduces fee churn)
# - Works in bull/bear: ATR regime adapts to volatility, filters low-vol chop, Donchian captures breakouts
# - Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_1w_donchian_atr_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 6h data
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 6h Donchian channel (20-period)
    donchian_upper = np.full(len(close_6h), np.nan)
    donchian_lower = np.full(len(close_6h), np.nan)
    donchian_mid = np.full(len(close_6h), np.nan)
    
    for i in range(19, len(close_6h)):
        if not (np.isnan(high_6h[i-19:i+1]).any() or np.isnan(low_6h[i-19:i+1]).any()):
            donchian_upper[i] = np.max(high_6h[i-19:i+1])
            donchian_lower[i] = np.min(low_6h[i-19:i+1])
            donchian_mid[i] = (donchian_upper[i] + donchian_lower[i]) / 2
    
    # Calculate 1w ATR(14)
    tr_1w = np.full(len(close_1w), np.nan)
    atr_1w = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):
        if not (np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i-1])):
            tr_1w[i] = max(
                high_1w[i] - low_1w[i],
                abs(high_1w[i] - close_1w[i-1]),
                abs(low_1w[i] - close_1w[i-1])
            )
    
    for i in range(13, len(tr_1w)):
        if not np.isnan(tr_1w[i-13:i+1]).any():
            atr_1w[i] = np.mean(tr_1w[i-13:i+1])
    
    # Calculate 1w ATR EMA(50)
    atr_ema_50_1w = np.full(len(atr_1w), np.nan)
    if len(atr_1w) >= 50:
        atr_ema_50_1w[49] = np.mean(atr_1w[:50])  # SMA seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(atr_1w)):
            if not np.isnan(atr_1w[i]):
                atr_ema_50_1w[i] = (atr_1w[i] - atr_ema_50_1w[i-1]) * multiplier + atr_ema_50_1w[i-1]
    
    # Calculate 6h volume EMA(20)
    volume_ema_20_6h = np.full(len(volume_6h), np.nan)
    if len(volume_6h) >= 20:
        volume_ema_20_6h[19] = np.mean(volume_6h[:20])  # SMA seed
        multiplier = 2 / (20 + 1)
        for i in range(20, len(volume_6h)):
            if not np.isnan(volume_6h[i]):
                volume_ema_20_6h[i] = (volume_6h[i] - volume_ema_20_6h[i-1]) * multiplier + volume_ema_20_6h[i-1]
    
    # Align all HTF/LTF indicators to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, prices, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, prices, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, prices, donchian_mid)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ema_50_1w)
    volume_ema_20_6h_aligned = align_htf_to_ltf(prices, prices, volume_ema_20_6h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(55, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(atr_ema_50_1w_aligned[i]) or np.isnan(volume_ema_20_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # ATR regime filter: ATR > 1.5x EMA = high volatility regime (trend follow)
        atr_regime = atr_1w_aligned[i] > 1.5 * atr_ema_50_1w_aligned[i]
        
        # Volume confirmation: current 6h volume > 1.5x 20-period EMA
        volume_confirm = volume_6h[i] > 1.5 * volume_ema_20_6h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper + ATR regime + volume confirmation
            if close_6h[i] > donchian_upper_aligned[i] and atr_regime and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + ATR regime + volume confirmation
            elif close_6h[i] < donchian_lower_aligned[i] and atr_regime and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to Donchian midpoint (mean reversion within channel)
            if position == 1:  # Long position
                if close_6h[i] <= donchian_mid_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_6h[i] >= donchian_mid_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals