#!/usr/bin/env python3
"""
1d_1w_Turtle_Breakout_Regime_v1
Hypothesis: On daily timeframe, buy breakouts above 20-day high with weekly uptrend filter and volume confirmation,
sell breakdowns below 20-day low with weekly downtrend and volume confirmation. Exit at opposite breakout level.
Uses weekly volatility regime filter to avoid choppy markets. Designed for low trade frequency (10-20/year) by requiring multiple confluence factors.
Works in bull/bear via weekly trend filter and mean-reversion exit at breakout levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Turtle_Breakout_Regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA(20) for trend direction
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).values
    weekly_uptrend = ema_20_1w > np.roll(ema_20_1w, 1)
    weekly_uptrend[0] = False  # first value has no previous
    weekly_downtrend = ema_20_1w < np.roll(ema_20_1w, 1)
    weekly_downtrend[0] = False
    
    # Weekly ATR(14) for volatility regime
    tr1_w = np.abs(high_1w[1:] - low_1w[:-1])
    tr2_w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))])
    atr_14_1w = np.zeros_like(tr_w)
    for i in range(len(tr_w)):
        if i < 14:
            atr_14_1w[i] = np.nan
        elif i == 14:
            atr_14_1w[i] = np.nanmean(tr_w[1:i+1])
        else:
            atr_14_1w[i] = (atr_14_1w[i-1] * 13 + tr_w[i]) / 14
    # Volatility regime: low volatility = trending market
    vol_ma_1w = np.zeros_like(atr_14_1w)
    for i in range(len(atr_14_1w)):
        if i < 30:
            vol_ma_1w[i] = np.nan
        else:
            vol_ma_1w[i] = np.mean(atr_14_1w[i-29:i+1])
    # Low volatility regime (trending) when current ATR < MA
    vol_regime_1w = atr_14_1w < vol_ma_1w
    
    # Align weekly data to daily timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    vol_regime_aligned = align_htf_to_ltf(prices, df_1w, vol_regime_1w.astype(float))
    
    # === DAILY BREAKOUT LEVELS ===
    # Donchian breakout: 20-day high/low
    high_20 = np.zeros(n)
    low_20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            high_20[i] = np.nan
            low_20[i] = np.nan
        else:
            high_20[i] = np.max(high[i-19:i+1])
            low_20[i] = np.min(low[i-19:i+1])
    
    # Daily volume average (20-period) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(ema_20_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_regime_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Only trade in low volatility (trending) regime on weekly
        in_trend_regime = vol_regime_aligned[i] > 0.5
        
        # Entry conditions
        long_setup = (close[i] > high_20[i]) and vol_confirm and in_trend_regime and weekly_uptrend_aligned[i] > 0.5
        short_setup = (close[i] < low_20[i]) and vol_confirm and in_trend_regime and weekly_downtrend_aligned[i] > 0.5
        
        # Exit conditions: mean reversion to opposite breakout level
        exit_long = close[i] < low_20[i]
        exit_short = close[i] > high_20[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals