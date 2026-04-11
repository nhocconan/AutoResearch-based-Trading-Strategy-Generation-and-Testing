#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter
# - Enter long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period volume SMA AND 1w close > 1w EMA20
# - Enter short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period volume SMA AND 1w close < 1w EMA20
# - Exit: Price returns to Camarilla Pivot level (mean reversion) OR opposite H3/L3 break
# - Camarilla levels derived from prior day's range provide institutional support/resistance
# - Volume confirmation ensures breakout validity
# - 1w EMA20 filter aligns with higher timeframe trend
# - Target: 25-40 trades/year to minimize fee drag while capturing high-probability breakouts

name = "4h_1d_1w_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla levels and volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 1w data ONCE before loop for trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d: based on prior day's range
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # Pivot = (high + low + close) / 3
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d
    camarilla_l3 = close_1d - 1.1 * range_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 4h timeframe (using completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Pre-compute volume SMA for 1d data (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute EMA20 for 1w close (trend filter)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Pre-compute 1w close aligned for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    for i in range(30, n):  # Start after 30-bar warmup for 20-period volume SMA
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        vol_confirm = volume_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1w close vs EMA20
        uptrend = close_1w_aligned[i] > ema_20_1w_aligned[i]
        downtrend = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        # Breakout signals
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = low[i] < camarilla_l3_aligned[i]
        
        # Mean reversion exit: price returns to pivot level
        long_exit = close[i] < camarilla_pivot_aligned[i]  # Exit long when price falls below pivot
        short_exit = close[i] > camarilla_pivot_aligned[i]  # Exit short when price rises above pivot
        
        # Trading logic
        if long_breakout and vol_confirm and uptrend:
            if position != 1:  # Only signal on new long entry
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif short_breakout and vol_confirm and downtrend:
            if position != -1:  # Only signal on new short entry
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:
            # Check for mean reversion exits
            if position == 1 and long_exit:
                position = 0
                signals[i] = 0.0
            elif position == -1 and short_exit:
                position = 0
                signals[i] = 0.0
            else:
                # Maintain current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals