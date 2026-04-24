#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and volume confirmation.
- Long when price breaks above H3 AND close > 4h EMA34 (bullish trend) AND volume > 1.5x ATR(14)*close
- Short when price breaks below L3 AND close < 4h EMA34 (bearish trend) AND volume > 1.5x ATR(14)*close
- Exit on trend reversal (close crosses 4h EMA34) or Camarilla mean reversion (price crosses H4/L4)
- Uses 1h primary timeframe with 4h HTF for trend direction to target 60-150 trades over 4 years (15-37/year)
- Camarilla pivot levels provide intraday support/resistance that work in ranging and trending markets
- 4h EMA34 ensures alignment with intermediate-term trend to avoid whipsaws in choppy/bear markets
- ATR-scaled volume filter adapts to changing volatility, reducing false breakouts
- Session filter (08-20 UTC) reduces noise from low-liquidity periods
- Designed for BTC/ETH with edge in bull markets (breakout continuation) and bear markets (trend filter prevents counter-trend entries)
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
    
    # Calculate Camarilla pivot levels (using previous day's OHLC)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #            L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # We need daily OHLC, so get 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    camarilla_mult = 1.1 / 2  # for H4/L4
    camarilla_mult_half = 1.1 / 4  # for H3/L3
    
    # Calculate levels from previous day (shift 1 to avoid look-ahead)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan  # first day has no previous
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    rang = prev_high_1d - prev_low_1d
    H4 = prev_close_1d + camarilla_mult * rang
    H3 = prev_close_1d + camarilla_mult_half * rang
    L3 = prev_close_1d - camarilla_mult_half * rang
    L4 = prev_close_1d - camarilla_mult * rang
    
    # Align daily Camarilla levels to 1h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Get 4h data ONCE before loop for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h EMA34 to 1h timeframe
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
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
    
    # Session filter: 08-20 UTC (avoid low-liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 34) + 1  # need at least 1d previous data and 4h EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(atr[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3, trend up (close > EMA34), volume confirmation, in session
            if close[i] > H3_aligned[i] and close[i] > ema_34_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below L3, trend down (close < EMA34), volume confirmation, in session
            elif close[i] < L3_aligned[i] and close[i] < ema_34_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend reverses (close < EMA34) OR price reaches L4 (mean reversion)
            if close[i] < ema_34_4h_aligned[i] or close[i] < L4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend reverses (close > EMA34) OR price reaches H4 (mean reversion)
            if close[i] > ema_34_4h_aligned[i] or close[i] > H4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34_VolumeConfirm_Session_v1"
timeframe = "1h"
leverage = 1.0