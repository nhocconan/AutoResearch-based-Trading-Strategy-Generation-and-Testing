#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Keltner Channel breakout with weekly EMA trend filter and volume confirmation.
# Uses 1d Keltner Channel (EMA20 ± ATR(10)*2) to identify breakouts.
# Long when price closes above upper KC with volume and weekly EMA up.
# Short when price closes below lower KC with volume and weekly EMA down.
# Designed to capture trends in both bull and bear markets by following weekly EMA.
# Keltner Channels adapt to volatility, reducing false breakouts in low volatility.
name = "1d_KeltnerChannel_WeeklyEMA_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Keltner Channel components
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    kc_upper = ema_20 + (2 * atr_10)
    kc_lower = ema_20 - (2 * atr_10)
    
    # Weekly EMA trend filter
    ema_weekly = pd.Series(df_weekly['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema_20[i]) or np.isnan(atr_10[i]) or np.isnan(ema_weekly_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price above upper KC + volume + weekly EMA up
            if (price > kc_upper[i] and vol_confirm[i] and price > ema_weekly_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below lower KC + volume + weekly EMA down
            elif (price < kc_lower[i] and vol_confirm[i] and price < ema_weekly_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below EMA20 or weekly EMA turns down
            if price < ema_20[i] or price < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above EMA20 or weekly EMA turns up
            if price > ema_20[i] or price > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals