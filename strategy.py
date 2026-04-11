#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly 13/26 EMA crossover on 1d timeframe with volume confirmation.
# Long when weekly EMA13 > EMA26 (bullish trend) and price breaks above daily Donchian(20) high with volume > 1.5x average.
# Short when weekly EMA13 < EMA26 (bearish trend) and price breaks below daily Donchian(20) low with volume > 1.5x average.
# Uses ATR(10) for volatility filter and exit signals.
# Designed for 15-30 trades/year on 1d timeframe with focus on trend continuation.
# Weekly EMA filter ensures alignment with higher timeframe trend, reducing counter-trend trades.
# Volume confirmation ensures institutional participation, reducing false breakouts.

name = "1d_1w_weekly_ema_volume_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    
    # Calculate weekly EMA(13) and EMA(26) for trend filter
    close_1w = df_1w['close'].values
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_26_1w = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Align weekly EMA to 1d timeframe
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    ema_26_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_26_1w)
    
    # Calculate Donchian channels (20-period) on 1d data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(10) for volatility filtering
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian period
        # Skip if any required data is invalid
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_13_1w_aligned[i]) or np.isnan(ema_26_1w_aligned[i]) or
            np.isnan(atr_10[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend direction
        is_bullish_trend = ema_13_1w_aligned[i] > ema_26_1w_aligned[i]
        is_bearish_trend = ema_13_1w_aligned[i] < ema_26_1w_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        bullish_breakout = (high[i] > high_max_20[i-1]) and vol_filter and is_bullish_trend
        bearish_breakout = (low[i] < low_min_20[i-1]) and vol_filter and is_bearish_trend
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on bearish breakout or bearish trend
            exit_long = bearish_breakout or (not is_bullish_trend)
        elif position == -1:
            # Exit short on bullish breakout or bullish trend
            exit_short = bullish_breakout or (not is_bearish_trend)
        
        # Priority: entry > exit > hold
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals