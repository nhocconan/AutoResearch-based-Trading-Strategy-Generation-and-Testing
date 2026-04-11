#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1-week volatility regime filter (ATR-based) and 1-week Donchian breakout.
# Uses weekly ATR to detect volatility regimes and weekly Donchian channels for trend direction.
# Long when price breaks above weekly Donchian high in high volatility regime, short when breaks below weekly Donchian low.
# Designed for low trade frequency (~10-30/year) to minimize fee decay while capturing major trend moves.
# Works in bull markets by capturing trends and in bear markets by avoiding false breakouts via volatility filter.

name = "1d_1w_donchian_atr_volatility_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly ATR(14) for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14 = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i == 0:
            atr_14[i] = np.nan
        elif i < 14:
            if i == 1:
                atr_14[i] = np.nanmean(tr[1:i+1])
            else:
                atr_14[i] = (atr_14[i-1] * (i-1) + tr[i]) / i
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate weekly Donchian channels (20-period)
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly indicators to daily timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    high_max_20_aligned = align_htf_to_ltf(prices, df_1w, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1w, low_min_20)
    
    # Calculate volatility regime: current ATR > 1.5 * ATR(50) average
    atr_50 = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 50:
            atr_50[i] = np.nan
        elif i == 50:
            atr_50[i] = np.nanmean(tr[1:51])
        else:
            atr_50[i] = (atr_50[i-1] * 49 + tr[i]) / 50
    
    atr_50_aligned = align_htf_to_ltf(prices, df_1w, atr_50)
    vol_regime = atr_14_aligned > (1.5 * atr_50_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure Donchian channels are valid
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: Donchian breakout in high volatility regime
        long_breakout = close[i] > high_max_20_aligned[i]
        short_breakout = close[i] < low_min_20_aligned[i]
        
        long_entry = long_breakout and vol_regime[i]
        short_entry = short_breakout and vol_regime[i]
        
        # Exit conditions: price returns to middle of Donchian channel
        donchian_mid = (high_max_20_aligned[i] + low_min_20_aligned[i]) / 2
        exit_long = close[i] < donchian_mid
        exit_short = close[i] > donchian_mid
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals