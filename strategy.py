#!/usr/bin/env python3
# Hypothesis: 6h Williams Fractal breakout with weekly trend filter and volume confirmation.
# Long when price breaks above latest weekly bullish fractal with 1w EMA34 uptrend and 6h volume > 1.5x 20-period average.
# Short when price breaks below latest weekly bearish fractal with 1w EMA34 downtrend and 6h volume > 1.5x 20-period average.
# Exit on opposite fractal level or at weekly open (WO).
# Uses 00-23 UTC session (full 6h coverage) to maximize signal reliability. Position size fixed at 0.25.
# Target: 50-150 trades over 4 years (12-37/year) for 6h timeframe.
# Works in bull/bear: 1w EMA34 ensures trend alignment, Williams Fractals provide structure within trend.
# Uses discrete position sizing to minimize fee churn and volume confirmation to reduce false breakouts.

name = "6h_WilliamsFractal_Breakout_1wEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (00-23 UTC - full coverage)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    # --- 6h Indicators (LTF) ---
    # 6h Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # 1w EMA34 for trend
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_bullish = close_1w > ema_34  # Bullish if price above EMA34
    ema_34_bearish = close_1w < ema_34  # Bearish if price below EMA34
    
    # Align 1w indicators to 6h
    ema_34_bullish_aligned = align_htf_to_ltf(prices, df_1w, ema_34_bullish.astype(float))
    ema_34_bearish_aligned = align_htf_to_ltf(prices, df_1w, ema_34_bearish.astype(float))
    
    # --- 1w Williams Fractals (HTF) ---
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    # Williams fractals need 2 extra 1w bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Weekly Open (WO) for exit reference
    weekly_open_aligned = align_htf_to_ltf(prices, df_1w, open_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if outside session or missing data
        if (not in_session[i] or
            np.isnan(ema_34_bullish_aligned[i]) or 
            np.isnan(ema_34_bearish_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(weekly_open_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above bullish fractal + 1w uptrend + volume confirmation
            if (close[i] > bullish_fractal_aligned[i] and 
                ema_34_bullish_aligned[i] > 0.5 and 
                volume_confirm[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below bearish fractal + 1w downtrend + volume confirmation
            elif (close[i] < bearish_fractal_aligned[i] and 
                  ema_34_bearish_aligned[i] > 0.5 and 
                  volume_confirm[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below bearish fractal OR price <= weekly open (WO)
            if close[i] < bearish_fractal_aligned[i] or close[i] <= weekly_open_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above bullish fractal OR price >= weekly open (WO)
            if close[i] > bullish_fractal_aligned[i] or close[i] >= weekly_open_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals