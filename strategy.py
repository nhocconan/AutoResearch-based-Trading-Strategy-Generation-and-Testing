#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ATR stop.
# Long when price breaks above Donchian(20) high with volume > 1.5x 20-bar avg.
# Short when price breaks below Donchian(20) low with volume > 1.5x 20-bar avg.
# Exit on opposite Donchian touch or ATR-based stop (2x ATR).
# Uses 1d EMA200 as trend filter: only long if price > EMA200, only short if price < EMA200.
# Designed for 4h timeframe to balance trade frequency and signal quality.
# Expected trades: 20-40 per year per symbol, avoiding excessive fee drag.
name = "4h_Donchian20_Volume_EMA200_ATRStop"
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
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR (14-period) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20, 14)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_200_val = ema_200_aligned[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        atr_val = atr[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above Donchian high, above EMA200, volume confirmed
            if price > donch_high_val and price > ema_200_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, below EMA200, volume confirmed
            elif price < donch_low_val and price < ema_200_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: exit on Donchian low touch or ATR stop
            exit_price = donch_low_val  # Exit if price touches Donchian low
            stop_price = ema_200_val - 2.0 * atr_val  # ATR-based stop below EMA200
            if price < exit_price or price < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position: exit on Donchian high touch or ATR stop
            exit_price = donch_high_val  # Exit if price touches Donchian high
            stop_price = ema_200_val + 2.0 * atr_val  # ATR-based stop above EMA200
            if price > exit_price or price > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals