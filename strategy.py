#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H4/L4 breakout with 1d EMA50 trend filter and volume confirmation
# Camarilla H4/L4 levels provide strong support/resistance for 12h breakouts
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
# Volume spike (>2.0x 20-period EMA) filters low-probability breakouts
# Designed for BTC/ETH in both bull/bear markets via trend filter + structure
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_Camarilla_H4L4_Breakout_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Shift to use previous bar's typical price (no look-ahead)
    typical_price_prev = np.roll(typical_price, 1)
    typical_price_prev[0] = np.nan  # First bar has no previous
    
    # Camarilla H4, L4 levels based on previous bar
    # H4 = 1.1/2 * (high - low) + close
    # L4 = close - 1.1/2 * (high - low)
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    camarilla_h4 = close_prev + 1.1/2 * (high_prev - low_prev)
    camarilla_l4 = close_prev - 1.1/2 * (high_prev - low_prev)
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start from 60 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Camarilla breakout signals with 1d trend filter
        # Long: Break above H4 + price above 1d EMA50 + volume spike
        # Short: Break below L4 + price below 1d EMA50 + volume spike
        if position == 0:
            if close[i] > camarilla_h4[i] and close[i] > ema_50_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            elif close[i] < camarilla_l4[i] and close[i] < ema_50_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below L4 (reversion to mean) OR below 1d EMA50
            if close[i] < camarilla_l4[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above H4 (reversion to mean) OR above 1d EMA50
            if close[i] > camarilla_h4[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals