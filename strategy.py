#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Bollinger Band breakout with volume confirmation and ATR stop.
# In both bull and bear markets, price tends to revert to the mean after touching
# Bollinger Bands, but strong breakouts with volume indicate momentum continuation.
# Uses weekly trend filter to only take breaks in the direction of the higher timeframe.
# Target: 20-50 trades over 4 years to minimize fee drag.

name = "1d_BollingerBreakout_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Daily ATR(14) for stop
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    tr1 = high_series - low_series
    tr2 = np.abs(high_series - close_series.shift())
    tr3 = np.abs(low_series - close_series.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[:1] = np.nan  # First TR is undefined
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need weekly EMA warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        weekly_trend = ema_34_1w_aligned[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: Break above upper BB with volume, only if weekly trend is up
            if price > bb_upper[i] and volume_confirmed and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower BB with volume, only if weekly trend is down
            elif price < bb_lower[i] and volume_confirmed and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below middle BB or ATR stop
            if price < bb_middle[i] or price < (high[i] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above middle BB or ATR stop
            if price > bb_middle[i] or price > (low[i] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals