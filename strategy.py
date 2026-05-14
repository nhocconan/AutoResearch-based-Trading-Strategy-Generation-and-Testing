#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and 12h volume spike confirmation.
# Long when price breaks above upper Donchian channel AND 12h EMA50 slope is positive AND 12h volume > 2.0 * 20-period average.
# Short when price breaks below lower Donchian channel AND 12h EMA50 slope is negative AND 12h volume > 2.0 * 20-period average.
# Exit when price retraces to the midpoint of the Donchian channel (mean reversion to structure).
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown. Target: 75-200 total trades over 4 years (19-50/year) for 4h.
# Works in both bull and bear markets: 12h EMA50 slope filter ensures we only trade in clear trending conditions,
# while volume confirmation avoids breakouts in low-participation environments. Donchian channels provide objective structure
# that adapts to volatility, reducing false breakouts and improving test generalization vs fixed pivot levels.

name = "4h_Donchian20_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # EMA50
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # EMA50 slope: bullish if current > previous, bearish if current < previous
    ema_50_slope = np.diff(ema_50, prepend=ema_50[0])
    ema_50_bullish = ema_50_slope > 0
    ema_50_bearish = ema_50_slope < 0
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume_12h > (2.0 * vol_ma_20_12h)
    
    # Align to 4h timeframe
    ema_50_bullish_aligned = align_htf_to_ltf(prices, df_12h, ema_50_bullish.astype(float))
    ema_50_bearish_aligned = align_htf_to_ltf(prices, df_12h, ema_50_bearish.astype(float))
    volume_confirm_aligned = align_htf_to_ltf(prices, df_12h, volume_confirm_12h.astype(float))
    
    # Calculate Donchian(20) channels on 4h data
    # Upper channel: 20-period high
    # Lower channel: 20-period low
    # Middle channel: midpoint (exit level)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_bullish_aligned[i]) or 
            np.isnan(ema_50_bearish_aligned[i]) or
            np.isnan(volume_confirm_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian channel AND bullish 12h EMA50 trend AND volume confirmation
            if (open_[i] <= highest_high[i] and close[i] > highest_high[i] and 
                ema_50_bullish_aligned[i] > 0.5 and 
                volume_confirm_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Donchian channel AND bearish 12h EMA50 trend AND volume confirmation
            elif (open_[i] >= lowest_low[i] and close[i] < lowest_low[i] and 
                  ema_50_bearish_aligned[i] > 0.5 and 
                  volume_confirm_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midpoint (mean reversion to structure)
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midpoint (mean reversion to structure)
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals