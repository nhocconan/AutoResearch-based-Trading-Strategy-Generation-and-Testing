#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot levels with 1-day EMA(100) filter and volume confirmation
# Long when price touches Camarilla L3 support, price > 1d EMA(100), and volume > 1.5x daily average
# Short when price touches Camarilla H3 resistance, price < 1d EMA(100), and volume > 1.5x daily average
# Exit when price crosses 1d EMA(100) or opposite pivot level is touched
# Stoploss at 1.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses Camarilla levels from daily timeframe for key support/resistance levels
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_camarilla_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for price reference
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 1d data for Camarilla calculation and EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (using previous day's data)
    # H4 = C + 1.5*(H-L), H3 = C + 1.0*(H-L), H2 = C + 0.5*(H-L), H1 = C
    # L4 = C - 1.5*(H-L), L3 = C - 1.0*(H-L), L2 = C - 0.5*(H-L), L1 = C
    # Where C = (H+L+C)/3 (typical price), but we'll use close for simplicity
    # Actually, standard Camarilla uses previous day's H, L, C
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First value will be NaN due to roll, handle it
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Calculate Camarilla levels
    diff = prev_high - prev_low
    H3 = prev_close + 1.0 * diff
    L3 = prev_close - 1.0 * diff
    
    # 1d EMA(100) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=100, adjust=False).mean().values
    
    # 1d volume average for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss (using 12h data)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d data to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 1.5 * ATR
            if close[i] < entry_price - 1.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses EMA or touches H3 (opposite level)
            elif close[i] >= ema_1d_aligned[i] or close[i] >= H3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 1.5 * ATR
            if close[i] > entry_price + 1.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses EMA or touches L3 (opposite level)
            elif close[i] <= ema_1d_aligned[i] or close[i] <= L3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price touches L3 support, price above EMA (bullish trend), volume spike
            if (close[i] <= L3_aligned[i] and
                close[i] > ema_1d_aligned[i] and
                volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price touches H3 resistance, price below EMA (bearish trend), volume spike
            elif (close[i] >= H3_aligned[i] and
                  close[i] < ema_1d_aligned[i] and
                  volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals