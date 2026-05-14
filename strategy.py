#!/usr/bin/env python3
# Hypothesis: 6h Williams Fractal breakout with 12h EMA trend filter and 1d volume confirmation.
# Long when price breaks above latest bearish Williams fractal with 12h EMA uptrend and 1d volume > 1.5x 20-period average.
# Short when price breaks below latest bullish Williams fractal with 12h EMA downtrend and 1d volume > 1.5x 20-period average.
# Exit on opposite fractal level (bullish fractal for longs, bearish for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe.
# Williams Fractals identify swing points; breakouts with HTF trend and volume confirmation capture momentum moves in both bull/bear markets.

name = "6h_WilliamsFractal_Breakout_12hEMA_Trend_1dVolumeConfirm_v1"
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
    
    # --- 6h Indicators (LTF) ---
    # 6h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_6h = volume > (1.5 * vol_ma_20)
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_trend_up = ema_50_12h > np.roll(ema_50_12h, 1)
    ema_trend_down = ema_50_12h < np.roll(ema_50_12h, 1)
    ema_trend_up[0] = False
    ema_trend_down[0] = False
    
    # Align 12h EMA trend to 6h
    ema_trend_up_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_up.astype(float))
    ema_trend_down_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_down.astype(float))
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # 1d volume confirmation: > 1.5x 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_20_1d)
    
    # Align 1d volume confirmation to 6h
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # --- Williams Fractals on 6h ---
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Needs 2 extra bars for confirmation (fractal forms at bar i, confirmed at i+2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_trend_up_aligned[i]) or 
            np.isnan(ema_trend_down_aligned[i]) or
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(volume_confirm_6h[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above bearish fractal + 12h EMA uptrend + 1d volume spike + 6h volume confirmation
            if (close[i] > bearish_fractal_aligned[i] and 
                ema_trend_up_aligned[i] > 0.5 and 
                volume_confirm_1d_aligned[i] > 0.5 and
                volume_confirm_6h[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below bullish fractal + 12h EMA downtrend + 1d volume spike + 6h volume confirmation
            elif (close[i] < bullish_fractal_aligned[i] and 
                  ema_trend_down_aligned[i] > 0.5 and 
                  volume_confirm_1d_aligned[i] > 0.5 and
                  volume_confirm_6h[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below bullish fractal
            if close[i] < bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above bearish fractal
            if close[i] > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals