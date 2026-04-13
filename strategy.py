#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h pivot-based trend detection and volume confirmation.
# Long: Price above 12h EMA(20) + 12h ADX > 25 + volume > 1.5x 20-period average.
# Short: Price below 12h EMA(20) + 12h ADX > 25 + volume > 1.5x 20-period average.
# Uses trend strength (ADX) to avoid whipsaws in ranging markets, EMA for direction, volume for confirmation.
# Time filter: None (all hours) to capture trends across sessions.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for EMA and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # EMA(20) on 12h close
    ema_20 = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 20:
        ema_20[19] = close_12h[:20].mean()
        for i in range(20, len(close_12h)):
            ema_20[i] = (close_12h[i] * 2 / (20 + 1)) + (ema_20[i-1] * (19 / (20 + 1)))
    
    # ADX(14) on 12h data
    adx = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 30:
        # True Range
        tr = np.zeros(len(close_12h))
        tr[0] = high_12h[0] - low_12h[0]
        for i in range(1, len(close_12h)):
            tr[i] = max(
                high_12h[i] - low_12h[i],
                abs(high_12h[i] - close_12h[i-1]),
                abs(low_12h[i] - close_12h[i-1])
            )
        
        # Directional Movement
        dm_plus = np.zeros(len(close_12h))
        dm_minus = np.zeros(len(close_12h))
        for i in range(1, len(close_12h)):
            up_move = high_12h[i] - high_12h[i-1]
            down_move = low_12h[i-1] - low_12h[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed values
        tr14 = np.zeros(len(close_12h))
        dm_plus14 = np.zeros(len(close_12h))
        dm_minus14 = np.zeros(len(close_12h))
        
        # Initial sums
        tr14[13] = tr[1:14].sum()
        dm_plus14[13] = dm_plus[1:14].sum()
        dm_minus14[13] = dm_minus[1:14].sum()
        
        # Wilder smoothing
        for i in range(14, len(close_12h)):
            tr14[i] = tr14[i-1] - (tr14[i-1] / 14) + tr[i]
            dm_plus14[i] = dm_plus14[i-1] - (dm_plus14[i-1] / 14) + dm_plus[i]
            dm_minus14[i] = dm_minus14[i-1] - (dm_minus14[i-1] / 14) + dm_minus[i]
        
        # Directional Indicators
        di_plus = np.zeros(len(close_12h))
        di_minus = np.zeros(len(close_12h))
        for i in range(14, len(close_12h)):
            if tr14[i] != 0:
                di_plus[i] = 100 * (dm_plus14[i] / tr14[i])
                di_minus[i] = 100 * (dm_minus14[i] / tr14[i])
        
        # DX and ADX
        dx = np.zeros(len(close_12h))
        for i in range(14, len(close_12h)):
            if (di_plus[i] + di_minus[i]) != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        
        # ADX: smoothed DX
        adx[27] = dx[14:28].mean()  # First ADX at index 27 (14+13)
        for i in range(28, len(close_12h)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 12h indicators to 6h
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_20_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema = ema_20_aligned[i]
        adx_val = adx_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Trend strength filter: ADX > 25
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long: price above EMA + strong trend + volume confirmation
            if (price > ema and strong_trend and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price below EMA + strong trend + volume confirmation
            elif (price < ema and strong_trend and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below EMA
            if price < ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above EMA
            if price > ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_EMA_ADX_Volume_Trend"
timeframe = "6h"
leverage = 1.0