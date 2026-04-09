#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d EMA trend filter and volume confirmation
# - Uses 1d Camarilla pivot levels (H3/L3) for breakout entries
# - Trend filter: 1d EMA(50) to align with daily trend and avoid counter-trend trades
# - Volume confirmation: volume > 2.0x 20-period average to ensure breakout strength
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~20-30 trades/year (80-120 total over 4 years) to stay under fee drag threshold
# - Works in BOTH bull markets (long breakouts at H3/H4) and bear markets (short breakdowns at L3/L4)
# - Daily EMA filter reduces whipsaws during counter-trend bounces

name = "4h_1d_camarilla_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3/L3 are the key breakout levels
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h volume > 2.0x 20-period average (strict volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high
            if low[i] <= highest_since_entry - (2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low
            if high[i] >= lowest_since_entry + (2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and 1d EMA trend filter
            # Long: price breaks above Camarilla H3 AND price > 1d EMA50 AND volume spike
            if (high[i] >= camarilla_h3_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and
                volume_spike[i]):
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            # Short: price breaks below Camarilla L3 AND price < 1d EMA50 AND volume spike
            elif (low[i] <= camarilla_l3_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  volume_spike[i]):
                position = -1
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals