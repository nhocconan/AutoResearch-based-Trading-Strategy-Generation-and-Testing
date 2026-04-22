#!/usr/bin/env python3

"""
Hypothesis: Daily Keltner Channel breakout with weekly trend filter and volume confirmation.
Only trade long when price breaks above upper Keltner Channel during low volatility (ATR-based)
and weekly trend is up; short when price breaks below lower Keltner Channel during low volatility
and weekly trend is down. Uses ATR-based channel width to identify low volatility periods,
avoiding false breakouts in high volatility periods. Designed for low trade frequency
(7-25 trades/year) by requiring multiple confirmations: low volatility, price breakout,
and trend alignment. Works in both bull and bear markets by following the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Keltner Channel (20, 2) on daily
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate True Range
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR (20-period)
    atr = tr.rolling(window=20, min_periods=20).mean().values
    
    # EMA (20-period) for middle line
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr
    
    # Keltner Channel width normalized by price for volatility measurement
    kc_width = (kc_upper - kc_lower) / close
    kc_width_pct = pd.Series(kc_width).rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Weekly EMA34 for trend direction
    weekly_close = df_weekly['close'].values
    ema34_weekly = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kc_width_pct[i]) or np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(ema34_weekly_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Low volatility condition: Keltner Channel width in lower 30th percentile
        low_vol = kc_width_pct[i] < 0.3
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: low volatility + price breaks above upper channel + weekly uptrend + volume spike
            if low_vol and close[i] > kc_upper[i] and ema34_weekly_aligned[i] > ema34_weekly_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: low volatility + price breaks below lower channel + weekly downtrend + volume spike
            elif low_vol and close[i] < kc_lower[i] and ema34_weekly_aligned[i] < ema34_weekly_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: volatility increase (end of low vol) or price returns to middle line
            exit_signal = False
            
            if position == 1:
                # Exit long: volatility increase or price closes below middle line
                if kc_width_pct[i] > 0.7 or close[i] < ema20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: volatility increase or price closes above middle line
                if kc_width_pct[i] > 0.7 or close[i] > ema20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_Keltner_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0