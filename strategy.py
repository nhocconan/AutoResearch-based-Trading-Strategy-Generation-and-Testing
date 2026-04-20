#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation.
# Donchian channels identify breakouts above/below recent 20-period highs/lows, capturing momentum.
# 1d EMA200 ensures we only trade in direction of higher timeframe trend (avoid counter-trend).
# Volume confirmation (volume > 1.5x 20-period average) filters weak breakouts.
# Target: 15-25 trades per year to minimize fee drift and work in both bull and bear markets.

name = "12h_Donchian20_1dEMA200_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for EMA200 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d EMA200 for trend direction ===
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # === 12h Donchian(20) breakout ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high: max of last 20 highs (excluding current bar)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    # Donchian low: min of last 20 lows (excluding current bar)
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 12h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after Donchian warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_200_aligned[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(donch_high_val) or 
            np.isnan(donch_low_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Donchian high + uptrend + volume
            if close_val > donch_high_val and close_val > ema_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short breakout: price breaks below Donchian low + downtrend + volume
            elif close_val < donch_low_val and close_val < ema_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend reversal
            if close_val < donch_low_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend reversal
            if close_val > donch_high_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals