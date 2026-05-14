#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (price > SMA50) and 6h volume confirmation (>1.5x 20-period average).
# Long when price breaks above 20-bar high with 1d price > SMA50 and volume confirmation.
# Short when price breaks below 20-bar low with 1d price < SMA50 and volume confirmation.
# Exit on opposite Donchian level or when trend filter fails.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe.
# Works in bull/bear: 1d SMA50 trend filter ensures alignment with higher timeframe direction, reducing false breakouts in ranging markets.

name = "6h_Donchian20_Breakout_1dSMA50_Trend_VolumeConfirm"
timeframe = "6h"
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
    
    # --- 6h Indicators (LTF) ---
    # 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_6h = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d SMA50 for trend filter
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Trend filter: price above/below SMA50
    trend_up = close > sma_50_1d_aligned  # Uptrend: price > SMA50
    trend_down = close < sma_50_1d_aligned  # Downtrend: price < SMA50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_confirm_6h[i]) or
            np.isnan(sma_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + uptrend + volume confirmation
            if (close[i] > donchian_high[i] and 
                trend_up[i] and 
                volume_confirm_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + downtrend + volume confirmation
            elif (close[i] < donchian_low[i] and 
                  trend_down[i] and 
                  volume_confirm_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or trend fails
            if (close[i] < donchian_low[i] or 
                not trend_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or trend fails
            if (close[i] > donchian_high[i] or 
                not trend_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals