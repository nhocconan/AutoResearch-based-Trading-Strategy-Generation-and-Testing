#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR stoploss.
# Long when price breaks above Donchian high (20) with price > 1d EMA50 and volume > 1.3x average.
# Short when price breaks below Donchian low (20) with price < 1d EMA50 and volume > 1.3x average.
# Exit on opposite Donchian break or ATR-based stop (2x ATR).
# Designed for 4h timeframe to capture medium-term trends in both bull and bear markets.
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years).
name = "4h_Donchian20_1dEMA50_Volume_ATRStop"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d EMA50 to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50, 14)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50_val = ema_50_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.3 * vol_ma
        
        if position == 0:
            # Enter long if price breaks above Donchian high, above EMA50, and volume confirmation
            if price > donch_high[i] and price > ema_50_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Enter short if price breaks below Donchian low, below EMA50, and volume confirmation
            elif price < donch_low[i] and price < ema_50_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: price breaks below Donchian low OR ATR stoploss
            if price < donch_low[i] or price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: price breaks above Donchian high OR ATR stoploss
            if price > donch_high[i] or price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals