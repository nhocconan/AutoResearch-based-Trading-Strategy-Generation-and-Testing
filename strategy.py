#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_ATRFilter_v2
Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volatility filter.
- Long when price breaks above 20-period high AND 1d EMA50 uptrend AND ATR(14) > ATR_ma(50) (high volatility regime)
- Short when price breaks below 20-period low AND 1d EMA50 downtrend AND ATR(14) > ATR_ma(50)
- Uses 6h chart for Donchian channels and entry timing
- 1d EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- ATR volatility filter ensures we only trade during sufficient volatility (avoids low-vol chop)
- Designed for low frequency (target 12-30 trades/year on 6h) to minimize fee drag
- Novelty: Combines Donchian breakout with volatility regime filter (trade only when ATR > its MA) to avoid false breakouts in low-vol environments
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
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter (needs completed 1d candle)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_1d = np.where(ema_50_1d_aligned > 0, 
                        np.where(close > ema_50_1d_aligned, 1, -1), 
                        0)
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR moving average for volatility regime filter
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    # Volatility filter: 1 = high volatility (ATR > ATR_MA), 0 = low volatility
    vol_filter = np.where(atr > atr_ma, 1, 0)
    
    # Calculate Donchian channels on 6h chart (primary timeframe)
    # Using 20-period lookback
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA, 50 for ATR MA, 20 for Donchian)
    start_idx = max(50, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(trend_1d[i]) or np.isnan(vol_filter[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with trend and volatility filter
        if position == 0:
            # Long: Price breaks above Donchian high AND 1d uptrend AND high volatility regime
            if close[i] > donchian_high[i] and trend_1d[i] == 1 and vol_filter[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND 1d downtrend AND high volatility regime
            elif close[i] < donchian_low[i] and trend_1d[i] == -1 and vol_filter[i] == 1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian low OR 1d trend turns down
            if close[i] < donchian_low[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian high OR 1d trend turns up
            if close[i] > donchian_high[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_ATRFilter_v2"
timeframe = "6h"
leverage = 1.0