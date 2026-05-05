#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation + ATR trailing stop
# Long when price breaks above Donchian upper(20) AND close > EMA34(1d) AND volume > 2.0x 20-period average
# Short when price breaks below Donchian lower(20) AND close < EMA34(1d) AND volume > 2.0x 20-period average
# Exit when price retraces to Donchian midpoint OR EMA34(1d) trend flip
# ATR-based trailing stop: exit long if price < highest_high_since_entry - 2.5*ATR(20)
# Uses 4h primary timeframe with 1d HTF for trend filter to reduce whipsaw and avoid overtrading
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-150 total trades over 4 years (19-37/year) to avoid fee drag
# Donchian channels provide clear structure; breakouts with volume and trend filter capture strong moves with controlled frequency

name = "4h_Donchian20_Breakout_1dEMA34_Trend_Volume_ATRstop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(20) for volatility and trailing stop
    if len(close) >= 20:
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    else:
        atr = np.full(n, np.nan)
    
    # Calculate Donchian channels (20-period)
    if len(high) >= 20 and len(low) >= 20:
        donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donch_mid = (donch_high + donch_low) / 2.0
    else:
        donch_high = np.full(n, np.nan)
        donch_low = np.full(n, np.nan)
        donch_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND close > EMA34(1d) AND volume spike
            if (high[i] > donch_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry = high[i]
            # Short conditions: price breaks below Donchian lower AND close < EMA34(1d) AND volume spike
            elif (low[i] < donch_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                lowest_since_entry = low[i]
        elif position == 1:
            # Update highest high since entry for trailing stop
            highest_since_entry = max(highest_since_entry, high[i])
            
            # Exit long: price retraces to Donchian midpoint OR close < EMA34(1d) (trend flip) OR ATR trailing stop
            trailing_stop = highest_since_entry - (2.5 * atr[i])
            if (close[i] <= donch_mid[i] or 
                close[i] < ema_34_1d_aligned[i] or 
                close[i] < trailing_stop):
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry for trailing stop
            lowest_since_entry = min(lowest_since_entry, low[i])
            
            # Exit short: price retraces to Donchian midpoint OR close > EMA34(1d) (trend flip) OR ATR trailing stop
            trailing_stop = lowest_since_entry + (2.5 * atr[i])
            if (close[i] >= donch_mid[i] or 
                close[i] > ema_34_1d_aligned[i] or 
                close[i] > trailing_stop):
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals