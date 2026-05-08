#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h trend filter (EMA34), 1d trend filter (EMA34), and volume confirmation on breakouts.
# Long when 4h trend up, 1d trend up, price breaks above 1h Donchian upper band, volume > 1.5x average.
# Short when 4h trend down, 1d trend down, price breaks below 1h Donchian lower band, volume > 1.5x average.
# Exit when any trend fails or price crosses opposite Donchian band.
# Uses multi-timeframe alignment to avoid look-ahead and ensure proper timing.
# Position size fixed at 0.20 to manage risk and reduce overtrading.
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag.

name = "1h_4h1dTrend_1hDonchian_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Get 1h data for Donchian bands (use same timeframe as price)
    # We'll calculate Donchian from price data directly
    
    # 4h EMA(34) for trend
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_4h_up = ema_34_4h > np.roll(ema_34_4h, 1)
    trend_4h_up = np.where(np.isnan(trend_4h_up), False, trend_4h_up)
    
    # 1d EMA(34) for trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = ema_34_1d > np.roll(ema_34_1d, 1)
    trend_1d_up = np.where(np.isnan(trend_1d_up), False, trend_1d_up)
    
    # 1h Donchian(20) bands from price data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    # Align 1d trend to 1h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_1d_up_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h trend up, 1d trend up, price breaks above 1h Donchian upper band, volume spike
            if (trend_4h_up_aligned[i] and trend_1d_up_aligned[i] and
                close[i] > donchian_high[i] and vol_ratio[i] > 1.5):
                signals[i] = 0.20
                position = 1
            # Short: 4h trend down, 1d trend down, price breaks below 1h Donchian lower band, volume spike
            elif (not trend_4h_up_aligned[i] and not trend_1d_up_aligned[i] and
                  close[i] < donchian_low[i] and vol_ratio[i] > 1.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: any trend fails or price breaks below Donchian lower band
            if (not trend_4h_up_aligned[i] or not trend_1d_up_aligned[i] or close[i] < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: any trend fails or price breaks above Donchian upper band
            if (trend_4h_up_aligned[i] or trend_1d_up_aligned[i] or close[i] > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals