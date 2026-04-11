#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_ema_crossover_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA for trend direction (21 and 55 period)
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_55_12h = pd.Series(close_12h).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema_21_4h = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    ema_55_4h = align_htf_to_ltf(prices, df_12h, ema_55_12h)
    
    # 4h ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h EMA crossover (9 and 21 period)
    ema_9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_21_4h[i]) or np.isnan(ema_55_4h[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_9[i]) or np.isnan(ema_21[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.5x average)
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: 12h EMA21 > EMA55 for long, EMA21 < EMA55 for short
        trend_long = ema_21_4h[i] > ema_55_4h[i]
        trend_short = ema_21_4h[i] < ema_55_4h[i]
        
        # EMA crossover signals
        ema_cross_up = ema_9[i] > ema_21[i] and ema_9[i-1] <= ema_21[i-1]
        ema_cross_down = ema_9[i] < ema_21[i] and ema_9[i-1] >= ema_21[i-1]
        
        # Long conditions: EMA crossover up + volume + 12h uptrend
        long_signal = volume_confirmed and trend_long and ema_cross_up
        
        # Short conditions: EMA crossover down + volume + 12h downtrend
        short_signal = volume_confirmed and trend_short and ema_cross_down
        
        # Exit when EMA crossover reverses
        exit_long = position == 1 and ema_9[i] < ema_21[i]
        exit_short = position == -1 and ema_9[i] > ema_21[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h EMA crossover strategy with 12h EMA trend filter and volume confirmation.
# Uses 12h EMA(21,55) for trend direction to filter 4h EMA(9,21) crossovers.
# Enters long when 4h EMA9 crosses above EMA21 with volume >1.5x average and 12h EMA21 > EMA55.
# Enters short when 4h EMA9 crosses below EMA21 with volume >1.5x average and 12h EMA21 < EMA55.
# Exits when EMA crossover reverses.
# Volume confirmation reduces false signals; 12h trend filter avoids counter-trend trades.
# Targets 20-40 trades/year to minimize fee drag while maintaining edge.
# Position size: 0.25 for balanced risk/return.
# Works in both bull and bear markets by following higher timeframe trend.