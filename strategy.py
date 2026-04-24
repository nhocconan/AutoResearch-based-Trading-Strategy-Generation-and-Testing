#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation.
- Long when price breaks above Camarilla H3 level AND close > 1d EMA34 (bullish trend)
- Short when price breaks below Camarilla L3 level AND close < 1d EMA34 (bearish trend)
- Volume must be > 1.5 * ATR(14) * close (volatility-adjusted volume filter)
- Exit on trend reversal (close crosses EMA34) or Camarilla mean reversion (price touches H4/L4)
- Uses 12h primary timeframe with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Camarilla levels provide precise intraday support/resistance that work in ranging and trending markets
- 1d EMA34 ensures alignment with long-term trend to avoid whipsaws in choppy/ bear markets
- ATR-scaled volume filter adapts to changing volatility, reducing false breakouts
- Designed for BTC/ETH with edge in both bull (breakout continuation) and bear (mean reversion at extremes) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (H3, L3, H4, L4) from previous 12h bar (no look-ahead)
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Range = high - low
    rng = high - low
    # Camarilla levels based on previous bar
    typical_price_prev = np.roll(typical_price, 1)
    rng_prev = np.roll(rng, 1)
    typical_price_prev[0] = np.nan
    rng_prev[0] = np.nan
    
    H3 = typical_price_prev + rng_prev * 1.1 / 4.0
    L3 = typical_price_prev - rng_prev * 1.1 / 4.0
    H4 = typical_price_prev + rng_prev * 1.1 / 2.0
    L4 = typical_price_prev - rng_prev * 1.1 / 2.0
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic volume threshold
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic volume threshold: volume > 1.5 * ATR * close (volatility-adjusted)
    vol_threshold = 1.5 * atr * close
    volume_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 34, 14) + 1  # Camarilla needs 1 bar, EMA34 needs 34, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(H4[i]) or np.isnan(L4[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3, trend up (close > EMA34), volume confirmation
            if close[i] > H3[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3, trend down (close < EMA34), volume confirmation
            elif close[i] < L3[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend reverses (close < EMA34) OR price touches H4 (mean reversion)
            if close[i] < ema_34_1d_aligned[i] or close[i] >= H4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reverses (close > EMA34) OR price touches L4 (mean reversion)
            if close[i] > ema_34_1d_aligned[i] or close[i] <= L4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0