#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy with 4h trend filter (EMA21) and 1d volatility filter (ATR-based range).
# Long when: price > 4h EMA21 AND price > 1d open + 0.5*ATR(1d) AND volume > 1.5x average volume.
# Short when: price < 4h EMA21 AND price < 1d open - 0.5*ATR(1d) AND volume > 1.5x average volume.
# Exit when price crosses 4h EMA21 in opposite direction.
# Uses 4h for trend direction, 1d for volatility context, 1h for entry timing.
# Session filter: 08-20 UTC to avoid low-volume periods.
# Position size: 0.20 (20%).
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate EMA21 on 4h
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR on 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr_1d = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Volatility bands: 1d open ± 0.5*ATR
    upper_band = open_1d + 0.5 * atr_1d
    lower_band = open_1d - 0.5 * atr_1d
    
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(21, 14, 20)  # Need EMA, ATR, and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(upper_band_aligned[i]) or
            np.isnan(lower_band_aligned[i]) or
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for entries
            # Long: price > 4h EMA21 AND price > 1d upper band AND volume confirmed
            if (close[i] > ema_4h_aligned[i] and 
                close[i] > upper_band_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price < 4h EMA21 AND price < 1d lower band AND volume confirmed
            elif (close[i] < ema_4h_aligned[i] and 
                  close[i] < lower_band_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 4h EMA21
            if close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 4h EMA21
            if close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4hEMA21_1dATRBands_Volume_Session"
timeframe = "1h"
leverage = 1.0