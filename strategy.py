#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 144-bar KAMA trend with 1d Bollinger squeeze filter
# - Uses KAMA (Kaufman Adaptive MA) to capture trend with low lag
# - Bollinger Band width < 20th percentile identifies low volatility (squeeze)
# - Breakouts from squeeze with trend alignment yield explosive moves
# - Works in both bull/bear: squeeze precedes major moves in any direction
# - Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# - Position size: 0.25 (25%) to balance return and drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands (20, 2.0) on daily close
    bb_period = 20
    bb_std = 2.0
    sma_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_1d + (bb_std * std_1d)
    lower_bb = sma_1d - (bb_std * std_1d)
    bb_width = upper_bb - lower_bb
    
    # Bollinger Band Width percentile (20-period lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Bollinger squeeze: width < 20th percentile
    squeeze = bb_width_percentile < 0.20
    
    # Align squeeze signal to 12h timeframe
    squeeze_12h = align_htf_to_ltf(prices, df_1d, squeeze)
    
    # Get 12h data for KAMA (trend filter)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate KAMA (144-period, fast=2, slow=30)
    kama_period = 144
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate efficiency ratio
    change = np.abs(np.diff(close_12h, kama_period))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)
    er = np.zeros_like(close_12h)
    er[kama_period:] = change[kama_period:] / volatility[kama_period:]
    er[:kama_period] = 0
    
    # Calculate smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_12h, np.nan)
    kama[kama_period] = close_12h[kama_period]
    for i in range(kama_period + 1, len(close_12h)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to 12h timeframe (already on 12h, but ensure alignment)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * volume_ma20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(200, kama_period + 50)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(squeeze_12h[i]) or 
            np.isnan(kama_12h_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > kama_12h_aligned[i]
        breakout_down = close[i] < kama_12h_aligned[i]
        
        if position == 0:
            # Enter long: squeeze + breakout up + volume
            if squeeze_12h[i] and breakout_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: squeeze + breakout down + volume
            elif squeeze_12h[i] and breakout_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: breakout down or squeeze ends
            if breakout_down or not squeeze_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout up or squeeze ends
            if breakout_up or not squeeze_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_BollingerSqueeze_Volume"
timeframe = "12h"
leverage = 1.0