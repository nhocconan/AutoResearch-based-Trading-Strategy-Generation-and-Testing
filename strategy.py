#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and 1w volume spike confirmation.
# Long when price breaks above Donchian upper band AND 1w EMA34 > EMA55 (bullish trend) AND 1w volume > 2.0 * 20-period average volume.
# Short when price breaks below Donchian lower band AND 1w EMA34 < EMA55 (bearish trend) AND 1w volume > 2.0 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Target: 30-100 total trades over 4 years (7-25/year) for 1d.
# Works in both bull and bear markets: 1w EMA crossover filter ensures we only trade in clear trending conditions,
# while volume confirmation avoids breakouts in low-participation environments.

name = "1d_Donchian20_Breakout_1wEMA34Trend_1wVolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate EMA34 and EMA55 on 1w timeframe
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema55_1w = pd.Series(close_1w).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Bullish trend: EMA34 > EMA55, Bearish trend: EMA34 < EMA55
    ema_trend_bullish = ema34_1w > ema55_1w
    ema_trend_bearish = ema34_1w < ema55_1w
    
    # Calculate 1w volume confirmation filter
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1w = volume_1w > (2.0 * vol_ma_20_1w)
    
    # Align to 1d timeframe
    ema_trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, ema_trend_bullish.astype(float))
    ema_trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, ema_trend_bearish.astype(float))
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm_1w.astype(float))
    
    # Calculate Donchian channel (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_upper[i] = np.nan
            donchian_lower[i] = np.nan
            donchian_mid[i] = np.nan
        else:
            period_high = np.max(high[i-20:i])
            period_low = np.min(low[i-20:i])
            donchian_upper[i] = period_high
            donchian_lower[i] = period_low
            donchian_mid[i] = (period_high + period_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_trend_bullish_aligned[i]) or 
            np.isnan(ema_trend_bearish_aligned[i]) or
            np.isnan(volume_confirm_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper band AND bullish 1w EMA trend AND volume confirmation
            if (open_[i] <= donchian_upper[i] and close[i] > donchian_upper[i] and 
                ema_trend_bullish_aligned[i] > 0.5 and 
                volume_confirm_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower band AND bearish 1w EMA trend AND volume confirmation
            elif (open_[i] >= donchian_lower[i] and close[i] < donchian_lower[i] and 
                  ema_trend_bearish_aligned[i] > 0.5 and 
                  volume_confirm_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals