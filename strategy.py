#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and 12h trend filter.
# Long: Price breaks above Camarilla H4 (resistance) + price > 12h EMA50 + volume > 1.5x avg volume (20-period).
# Short: Price breaks below Camarilla L4 (support) + price < 12h EMA50 + volume > 1.5x avg volume.
# Uses 12h for trend direction, 4h for pivot structure, volume for confirmation.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 20-50 total trades over 4 years (5-12.5/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 4h data for Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: H4, L4 (based on previous period)
    camarilla_h4 = np.full(len(close_4h), np.nan)
    camarilla_l4 = np.full(len(close_4h), np.nan)
    for i in range(1, len(close_4h)):
        # Use previous period's high, low, close
        ph = high_4h[i-1]
        pl = low_4h[i-1]
        pc = close_4h[i-1]
        camarilla_h4[i] = pc + 1.1 * (ph - pl) / 2
        camarilla_l4[i] = pc - 1.1 * (ph - pl) / 2
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 4h Camarilla to 4h (same timeframe, but need alignment for proper bar timing)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    
    # Align 12h EMA50 to 4h
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        ema_trend = ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: break above Camarilla H4 + above EMA50 + volume confirmation
            if (price > h4 and 
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below Camarilla L4 + below EMA50 + volume confirmation
            elif (price < l4 and 
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below Camarilla L4 or below EMA50
            if (price < l4 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above Camarilla H4 or above EMA50
            if (price > h4 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Camarilla_EMA_Volume"
timeframe = "4h"
leverage = 1.0