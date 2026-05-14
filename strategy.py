#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and 1w volume spike confirmation.
# Long when price breaks above 20-day high AND 1w close > 1w EMA50 (uptrend) AND 1w volume > 2.0 * 20-period average volume.
# Short when price breaks below 20-day low AND 1w close < 1w EMA50 (downtrend) AND 1w volume > 2.0 * 20-period average volume.
# Exit when price retraces to the midpoint of the 20-day Donchian channel.
# Uses discrete position sizing (0.30) to limit fee churn. Designed for 1d timeframe with strict entry conditions to avoid overtrading.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

name = "1d_Donchian20_Breakout_1wEMA50_1wVolumeSpike_v1"
timeframe = "1d"
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w volume confirmation filter (HTF)
    volume_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1w = volume_1w > (2.0 * vol_ma_20_1w)  # Volume > 2.0x 20-period MA
    volume_confirm_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm_1w.astype(float))
    
    # Calculate 20-day Donchian channel (based on prior completed day's OHLC)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)  # Midpoint for exit
    
    # For each 1d bar, use prior 20 completed days' OHLC
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        lookback_start = current_time - pd.Timedelta(days=20)
        
        day_mask = (df_1w['open_time'] >= lookback_start) & (df_1w['open_time'] < current_time)
        if day_mask.any() and day_mask.sum() >= 20:
            lookback_data = df_1w.loc[day_mask]
            donchian_high[i] = lookback_data['high'].max()
            donchian_low[i] = lookback_data['low'].min()
            donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
        else:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
            donchian_mid[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_confirm_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian high AND 1w close > 1w EMA50 (uptrend) AND volume confirmation
            if (open_[i] <= donchian_high[i] and close[i] > donchian_high[i] and 
                close_1w_values[i] > ema_50_1w_aligned[i] and 
                volume_confirm_1w_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # SHORT: price breaks below Donchian low AND 1w close < 1w EMA50 (downtrend) AND volume confirmation
            elif (open_[i] >= donchian_low[i] and close[i] < donchian_low[i] and 
                  close_1w_values[i] < ema_50_1w_aligned[i] and 
                  volume_confirm_1w_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Fix: close_1w_values needs to be defined before use in the loop
# Recalculate close_1w_values aligned to LTF
    close_1w = df_1w['close'].values
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)