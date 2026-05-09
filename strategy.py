# For 12h: test with 1d HTF for trend filter and volatility filter
# Strategy: 12h Donchian(20) breakout + 1d trend filter + volatility filter (ATR ratio)
# Entry: Price breaks above/below 12h Donchian channel with 1d EMA trend and ATR ratio < 0.8 (low volatility)
# Exit: Price returns to 12h Donchian midline or ATR ratio > 1.2 (high volatility)
# Position size: 0.25
# Expect ~15-25 trades/year per symbol, low frequency to avoid fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_Breakout_1dTrend_VolatilityFilter"
timezone = "UTC"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for trend filter and volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h Donchian channel (20 periods)
    # Calculate on 12h data directly
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR for volatility filter (14-period ATR on 12h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period ATR average (to detect low/high volatility regimes)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / np.where(atr_ma > 0, atr_ma, np.nan)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for ATR MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long entry: price breaks above Donchian high with daily uptrend and low volatility
            if (close[i] > donchian_high[i] and 
                close[i] > ema_34_12h[i] and  # daily uptrend
                atr_ratio[i] < 0.8 and      # low volatility regime
                in_session):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with daily downtrend and low volatility
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_34_12h[i] and  # daily downtrend
                  atr_ratio[i] < 0.8 and      # low volatility regime
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midline OR volatility expands
            if (close[i] < donchian_mid[i] or atr_ratio[i] > 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midline OR volatility expands
            if (close[i] > donchian_mid[i] or atr_ratio[i] > 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals