#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(34) trend filter and 1d volume confirmation
# - Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 1.5x 20-period average
# - Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 1.5x 20-period average
# - Exit when price crosses 1d EMA34 (trend reversal) or opposite Donchian band touch
# - Position size: 0.25 (25%) to balance return and drawdown
# - Designed for 4h timeframe to limit trades to 20-50/year, reducing fee drag
# - Uses higher timeframe (1d) for trend and volume to avoid whipsaws in both bull and bear markets

name = "4h_Donchian_EMA34_Volume_25"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 1d average volume
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + uptrend + volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_34_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + downtrend + volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema_34_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on trend reversal (price < EMA34) or touch opposite band
            if close[i] < ema_34_1d_aligned[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on trend reversal (price > EMA34) or touch opposite band
            if close[i] > ema_34_1d_aligned[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals