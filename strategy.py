#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_ATRFilter_v1
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based volatility filter.
- Long when price breaks above Donchian(20) high AND 1d EMA34 uptrend AND ATR(14) > ATR(50) (vol expansion)
- Short when price breaks below Donchian(20) low AND 1d EMA34 downtrend AND ATR(14) > ATR(50)
- Uses Donchian channels from completed 4h bars for structure-based breakouts
- 1d EMA34 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- ATR(14) > ATR(50) confirms volatility expansion (institutional participation)
- Designed for moderate frequency (target 19-50 trades/year) to minimize fee drag
- Exit on opposite Donchian level touch or trend reversal
- Novelty: Combines Donchian breakouts with HTF trend and volatility expansion filter for BTC/ETH edge in both bull/bear markets
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
    
    # Load 4h data ONCE before loop for Donchian levels (structure)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Donchian(20) levels from prior 4h bar (completed bar only)
    # Donchian high = max(high, lookback=20), low = min(low, lookback=20)
    lookback = 20
    donch_high = pd.Series(df_4h['high'].values).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(df_4h['low'].values).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align Donchian levels to 4h timeframe (no additional delay needed for structure)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Load daily data ONCE before loop for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA34 for trend filter (needs completed daily candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
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
            np.isnan(trend_1d[i]) or np.isnan(atr_expansion[i])):
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
            # Long: Price breaks above Donchian high AND daily uptrend AND vol expansion
            if close[i] > donch_high_aligned[i] and trend_1d[i] == 1 and atr_expansion[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND daily downtrend AND vol expansion
            elif close[i] < donch_low_aligned[i] and trend_1d[i] == -1 and atr_expansion[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian low OR daily trend turns down
            if close[i] < donch_low_aligned[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian high OR daily trend turns up
            if close[i] > donch_high_aligned[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0