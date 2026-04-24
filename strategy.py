#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
- Long when price breaks above Camarilla R1 AND close > 1d EMA34 (bullish trend)
- Short when price breaks below Camarilla S1 AND close < 1d EMA34 (bearish trend)
- Volume must be > 2.0 * ATR(14) (volatility-adjusted volume filter)
- Exit on trend reversal (close crosses 1d EMA34) or Donchian mean reversion (close crosses opposite Camarilla level)
- Uses 4h primary timeframe with 1d HTF to target 75-200 trades over 4 years (19-50/year)
- Camarilla levels provide precise intraday support/resistance that work in ranging markets
- 1d EMA34 ensures alignment with long-term trend to avoid whipsaws in bear markets
- ATR-scaled volume filter adapts to changing volatility, reducing false breakouts
- Designed for BTC/ETH with edge in bull markets (breakout continuation) and bear markets (avoiding false breakouts via trend filter)
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
    
    # Calculate Camarilla levels (based on previous day's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
    #            S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low), 
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We'll use daily OHLC to calculate these levels
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1 based on previous day's range
    daily_range = high_1d - low_1d
    camarilla_R1 = close_1d + 0.275 * daily_range
    camarilla_S1 = close_1d - 0.275 * daily_range
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Calculate ATR(14) for dynamic volume threshold
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic volume threshold: volume > 2.0 * ATR (volatility-adjusted)
    vol_threshold = 2.0 * atr
    volume_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 1) + 1  # 34 for EMA, 1 for Camarilla (needs previous day)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1, trend up (close > EMA34), volume confirmation
            if close[i] > camarilla_R1_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1, trend down (close < EMA34), volume confirmation
            elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below EMA34 (trend reversal) OR below Camarilla S1 (mean reversion)
            if close[i] < ema_34_1d_aligned[i] or close[i] < camarilla_S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above EMA34 (trend reversal) OR above Camarilla R1 (mean reversion)
            if close[i] > ema_34_1d_aligned[i] or close[i] > camarilla_R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA34_ATRVolConfirm_v1"
timeframe = "4h"
leverage = 1.0