#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot with 1d trend filter and volume confirmation.
# Camarilla levels provide strong support/resistance. Combined with 1d EMA trend filter
# and volume spikes, it filters false breakouts. Target: 12-37 trades/year (50-150 total)
# for 12h timeframe. Works in bull/bear via trend filter and mean-reversion at extremes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA(50) for 1d trend filter
    ema50_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (50 + 1)
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * ema_multiplier + ema50_1d[i-1]
    
    # Align 1d EMA to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from previous day
    camarilla_h4 = np.full(len(high_1d), np.nan)  # H4 resistance
    camarilla_l4 = np.full(len(low_1d), np.nan)   # L4 support
    camarilla_h3 = np.full(len(high_1d), np.nan)  # H3 resistance
    camarilla_l3 = np.full(len(low_1d), np.nan)   # L3 support
    
    for i in range(1, len(high_1d)):
        # Use previous day's data
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        rang = ph - pl
        
        if rang <= 0:
            camarilla_h4[i] = camarilla_l4[i] = camarilla_h3[i] = camarilla_l3[i] = pc
        else:
            camarilla_h4[i] = pc + 1.1 * rang / 2
            camarilla_l4[i] = pc - 1.1 * rang / 2
            camarilla_h3[i] = pc + 1.1 * rang / 4
            camarilla_l3[i] = pc - 1.1 * rang / 4
    
    # Align Camarilla levels to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Average volume (4-period = 2 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(4, n):
        avg_volume[i] = np.mean(volume[i-4:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(4, n):
        # Skip if any required data is not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_1d_aligned[i]
        
        camarilla_levels = {
            'h4': h4_aligned[i],
            'l4': l4_aligned[i],
            'h3': h3_aligned[i],
            'l3': l3_aligned[i]
        }
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: Price crosses above L3 with volume + above 1d EMA50
            if (price > camarilla_levels['l3'] and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Price crosses below H3 with volume + below 1d EMA50
            elif (price < camarilla_levels['h3'] and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price crosses below L4 or trend changes
            if (price < camarilla_levels['l4'] or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price crosses above H4 or trend changes
            if (price > camarilla_levels['h4'] or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_Trend_Volume"
timeframe = "12h"
leverage = 1.0