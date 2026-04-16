#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with 1-week EMA34 trend filter and volume confirmation
# Long when price breaks above 20-day high AND price > weekly EMA34 (uptrend) AND volume > 1.5x 20-day average
# Short when price breaks below 20-day low AND price < weekly EMA34 (downtrend) AND volume > 1.5x 20-day average
# Donchian channels capture breakouts, weekly EMA filters trend alignment, volume confirms conviction
# Discrete position sizing (0.25) to control drawdown. Target: 30-100 total trades over 4 years

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian and volume calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data once before loop for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Donchian Channel (20-period) ===
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # === 1w Indicator: EMA (34-period) for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to 1d timeframe
    highest_high_20_aligned = align_htf_to_ltf(prices, df_1d, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_20)
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-period 1d volume SMA
        vol_1d_series = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_series)
        vol_confirm = False
        if not np.isnan(vol_1d_aligned[i]):
            vol_threshold = vol_sma_20_1d_aligned[i] * 1.5
            vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # === LONG CONDITIONS ===
        # Price breaks above 20-day high AND price > weekly EMA34 (uptrend) AND volume confirmation
        if (close[i] > highest_high_20_aligned[i]) and (close[i] > ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Price breaks below 20-day low AND price < weekly EMA34 (downtrend) AND volume confirmation
        elif (close[i] < lowest_low_20_aligned[i]) and (close[i] < ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_1wEMA34_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0