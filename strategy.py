#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation.
# Donchian(20) captures breakouts in both bull and bear markets.
# 1d EMA50 filter ensures we trade in direction of higher timeframe trend.
# Volume spike (>1.5x 20-period average) confirms conviction.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
name = "4h_Donchian20_1dEMA50_Volume"
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
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Donchian channels on 4h data
    high_4h = pd.Series(df_4h['high'].values)
    low_4h = pd.Series(df_4h['low'].values)
    donchian_high = high_4h.rolling(window=20, min_periods=20).max().values
    donchian_low = low_4h.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to lower timeframe (4h) - no shift needed as we're on same TF
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA50 on 1d data
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to lower timeframe (4h)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema50 = ema_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high AND price above 1d EMA50 AND volume spike
            if price > upper and price > ema50 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND price below 1d EMA50 AND volume spike
            elif price < lower and price < ema50 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian low OR price below 1d EMA50
            if price < lower or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above Donchian high OR price above 1d EMA50
            if price > upper or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals