#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1w trend filter (EMA34), 4h Donchian breakout (20), and volume confirmation.
# Long when 1w trend up, price breaks above 4h Donchian upper band, volume > 1.5x average.
# Short when 1w trend down, price breaks below 4h Donchian lower band, volume > 1.5x average.
# Includes volatility-based exit (exit when volatility drops) to reduce whipsaw.
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag.
# Works in bull (trend follow) and bear (trend still exists in downtrends).

name = "4h_1wTrend_4hDonchian_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get 4h data for Donchian bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 1w EMA(34) for trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = ema_34_1w > np.roll(ema_34_1w, 1)
    trend_1w_up = np.where(np.isnan(trend_1w_up), False, trend_1w_up)
    
    # 4h Donchian(20) bands
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 1w trend to 4h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    # Align 4h Donchian bands to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # Volatility filter: ATR(14) / ATR(50) < 0.6 (low volatility regime)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50
    low_vol = atr_ratio < 0.6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1w trend up, price breaks above 4h Donchian upper band, volume spike, low volatility
            if (trend_1w_up_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 1.5 and
                low_vol[i]):
                signals[i] = 0.25
                position = 1
            # Short: 1w trend down, price breaks below 4h Donchian lower band, volume spike, low volatility
            elif (not trend_1w_up_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 1.5 and
                  low_vol[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend break or price breaks below Donchian lower band or volatility spikes
            if (not trend_1w_up_aligned[i] or close[i] < donchian_low_aligned[i] or not low_vol[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend break or price breaks above Donchian upper band or volatility spikes
            if (trend_1w_up_aligned[i] or close[i] > donchian_high_aligned[i] or not low_vol[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals