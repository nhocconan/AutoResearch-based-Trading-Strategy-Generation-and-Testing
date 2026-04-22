#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 12h Donchian breakout + volume confirmation.
# Uses 12h Donchian channel (20-period) for directional bias and 4h Choppiness Index (14-period) to filter trending markets.
# Long when price breaks above 12h Donchian high AND chop < 38.2 (trending).
# Short when price breaks below 12h Donchian low AND chop < 38.2 (trending).
# Volume confirmation reduces false breakouts.
# Designed for 4h timeframe with ~30-50 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Donchian channel (trend signal)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian channel: 20-period high/low (prior 12h)
    high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Shift to use only completed 12h bars (avoid look-ahead)
    high_20_12h = np.roll(high_20_12h, 1)
    low_20_12h = np.roll(low_20_12h, 1)
    
    # Load 4h data for Choppiness Index (regime filter)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range (TR)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no prior close)
    tr[0] = high_4h[0] - low_4h[0]
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(sumTR / (ATR * 14)) / log10(14)
    # Avoid division by zero or invalid values
    atr14 = atr * 14
    chop = np.full_like(close_4h, np.nan, dtype=float)
    mask = (sum_tr > 0) & (atr14 > 0)
    chop[mask] = 100 * np.log10(sum_tr[mask] / atr14[mask]) / np.log10(14)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    # Align indicators to 4h timeframe
    high_20_12h_aligned = align_htf_to_ltf(prices, df_12h, high_20_12h)
    low_20_12h_aligned = align_htf_to_ltf(prices, df_12h, low_20_12h)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_20_12h_aligned[i]) or np.isnan(low_20_12h_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian high + chop < 38.2 (trending) + volume spike
            if (close[i] > high_20_12h_aligned[i] and 
                chop_aligned[i] < 38.2 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian low + chop < 38.2 (trending) + volume spike
            elif (close[i] < low_20_12h_aligned[i] and 
                  chop_aligned[i] < 38.2 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price breaks opposite 12h Donchian level or chop > 61.8 (ranging)
            if position == 1:
                if (close[i] < low_20_12h_aligned[i] or chop_aligned[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > high_20_12h_aligned[i] or chop_aligned[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Choppiness_Regime_12hDonchian_Breakout_Volume_Spike"
timeframe = "4h"
leverage = 1.0