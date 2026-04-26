#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_ATRFilter_v1
Hypothesis: Daily Donchian(20) breakout with 1-week EMA34 trend filter and ATR-based volatility filter.
- Long when price breaks above Donchian(20) high AND 1w EMA34 uptrend AND ATR(14) > ATR(50) (vol expansion)
- Short when price breaks below Donchian(20) low AND 1w EMA34 downtrend AND ATR(14) > ATR(50)
- Uses Donchian channels from completed daily bars for structure-based breakouts
- 1-week EMA34 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- ATR(14) > ATR(50) confirms volatility expansion (institutional participation)
- Designed for low frequency (target 7-25 trades/year) to minimize fee drag on 1d timeframe
- Exit on opposite Donchian level touch or trend reversal
- Novelty: Combines Donchian breakouts with weekly trend and volatility expansion filter for BTC/ETH edge in both bull/bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load daily data ONCE before loop for Donchian levels (structure)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian(20) levels from prior daily bar (completed bar only)
    # Donchian high = max(high, lookback=20), low = min(low, lookback=20)
    lookback = 20
    donch_high = pd.Series(df_1d['high'].values).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(df_1d['low'].values).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align Donchian levels to daily timeframe (no additional delay needed for structure)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Load weekly data ONCE before loop for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter (needs completed weekly candle)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1w = np.where(ema_34_1w_aligned > 0, 
                        np.where(close > ema_34_1w_aligned, 1, -1), 
                        0)
    
    # Calculate ATR filter: ATR(14) > ATR(50) for volatility expansion
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_expansion = atr_14 > atr_50  # Volatility expansion filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for ATR, 34 for EMA, 20 for Donchian)
    start_idx = max(50, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(trend_1w[i]) or np.isnan(atr_expansion[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with trend and volatility expansion filter
        if position == 0:
            # Long: Price breaks above Donchian high AND weekly uptrend AND vol expansion
            if close[i] > donch_high_aligned[i] and trend_1w[i] == 1 and atr_expansion[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND weekly downtrend AND vol expansion
            elif close[i] < donch_low_aligned[i] and trend_1w[i] == -1 and atr_expansion[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian low OR weekly trend turns down
            if close[i] < donch_low_aligned[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian high OR weekly trend turns up
            if close[i] > donch_high_aligned[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_ATRFilter_v1"
timeframe = "1d"
leverage = 1.0