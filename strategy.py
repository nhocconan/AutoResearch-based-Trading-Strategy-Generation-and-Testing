#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    Strategy: 1D Weekly Donchian Breakout with Volume Confirmation and ATR Stop
    Hypothesis: Weekly Donchian(20) breakouts on daily chart with volume confirmation
    capture major trend moves. Works in bull/bear by capturing breakouts in either direction.
    Weekly filter ensures only strong breaks trigger, reducing false signals and trade frequency.
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian channel (higher timeframe filter)
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly Donchian channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian(20) - using rolling window
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation (20-day average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss and position sizing reference (14-day)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Daily close breaks above weekly Donchian high with volume confirmation
            if close[i] > donchian_high_aligned[i] and volume[i] > 1.5 * vol_ma20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Daily close breaks below weekly Donchian low with volume confirmation
            elif close[i] < donchian_low_aligned[i] and volume[i] > 1.5 * vol_ma20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: ATR-based stop loss or Donchian reversal
            if position == 1:
                # Long exit: price drops below weekly Donchian low OR 2*ATR stop from entry
                if close[i] < donchian_low_aligned[i] or close[i] < (signals[i-1] * atr[i] * 2):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Short exit: price rises above weekly Donchian high OR 2*ATR stop from entry
                if close[i] > donchian_high_aligned[i] or close[i] > (signals[i-1] * atr[i] * 2):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_Weekly_Donchian_Breakout_Volume_ATR_Stop_v1"
timeframe = "1d"
leverage = 1.0