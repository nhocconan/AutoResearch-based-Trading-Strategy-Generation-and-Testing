#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_breakout_v4"
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
    
    # Get weekly data for Keltner calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate ATR on weekly data (period=10)
    atr_weekly = np.zeros_like(close_weekly)
    tr_weekly = np.maximum(
        high_weekly[1:] - low_weekly[1:],
        np.maximum(
            np.abs(high_weekly[1:] - close_weekly[:-1]),
            np.abs(low_weekly[1:] - close_weekly[:-1])
        )
    )
    atr_weekly[10:] = pd.Series(tr_weekly).rolling(window=10, min_periods=10).mean().values
    atr_weekly[:10] = np.nan
    
    # Calculate EMA on weekly data (period=20)
    ema_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner channels on weekly data
    upper_weekly = ema_weekly + (2 * atr_weekly)
    lower_weekly = ema_weekly - (2 * atr_weekly)
    
    # Align Keltner channels to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_weekly, upper_weekly)
    lower_aligned = align_htf_to_ltf(prices, df_weekly, lower_weekly)
    ema_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Volume filter - 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price breaks above upper Keltner band with volume confirmation
        long_signal = close[i] > upper_aligned[i] and volume_ok[i]
        # Short: price breaks below lower Keltner band with volume confirmation
        short_signal = close[i] < lower_aligned[i] and volume_ok[i]
        
        # Exit when price returns to EMA (middle line)
        exit_long = close[i] < ema_aligned[i]
        exit_short = close[i] > ema_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals