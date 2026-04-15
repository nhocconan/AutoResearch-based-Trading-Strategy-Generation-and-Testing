#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + daily volume confirmation + ATR stoploss
# Uses daily Donchian channels for trend direction, volume > 1.5x 20-bar median for confirmation,
# and ATR-based exits. Designed for low-frequency trading (target: 20-50 trades/year) to minimize
# fee drag while capturing major trends in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Daily ATR for volatility filtering and stoploss
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[0], tr_1d])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: current volume > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above Donchian high + volume confirmation
        if (close[i] > donchian_high_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low + volume confirmation
        elif (close[i] < donchian_low_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: ATR-based trailing stop
        else:
            if i > 0 and signals[i-1] != 0:
                if signals[i-1] > 0:  # Long position
                    if close[i] < donchian_high_aligned[i] - 1.5 * atr_1d_aligned[i]:
                        signals[i] = 0.0
                    else:
                        signals[i] = signals[i-1]
                else:  # Short position
                    if close[i] > donchian_low_aligned[i] + 1.5 * atr_1d_aligned[i]:
                        signals[i] = 0.0
                    else:
                        signals[i] = signals[i-1]
            else:
                signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_Breakout_Volume_ATR"
timeframe = "12h"
leverage = 1.0