#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d EMA trend filter and volume confirmation
# - Long: price breaks above Camarilla H3 level, volume > 1.8x 20-bar avg, price > 1d EMA(50)
# - Short: price breaks below Camarilla L3 level, volume > 1.8x 20-bar avg, price < 1d EMA(50)
# - Exit: price returns to Camarilla pivot (midpoint) or opposite H3/L3 level
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 25-40 trades/year (100-160 total over 4 years) to stay within fee drag limits
# - Camarilla levels derived from prior day's range work well in both trending and ranging markets
# - Volume confirmation ensures breakouts have conviction
# - 1d EMA filter provides higher timeframe trend bias to avoid counter-trend trades

name = "4h_1d_camarilla_pivot_breakout_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for pivot calculation and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute Camarilla pivot levels from 1d OHLC
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), etc.
    # L4 = close - 1.5*(high-low), L3 = close - 1.1*(high-low), etc.
    # Pivot = (high + low + close) / 3
    
    # Shift 1d data by 1 to use previous day's OHLC for today's levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # H3 and L3 are the key levels for breakout trades
    h3 = pivot + 1.1 * range_hl
    l3 = pivot - 1.1 * range_hl
    
    # Align HTF levels to LTF (these levels are valid for the entire 4h bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        pivot_level = pivot_aligned[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume_current > 1.8 * volume_sma_20[i]
        
        # 1d EMA trend bias
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above H3, volume confirmation, long bias
        if close_price > h3_level and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below L3, volume confirmation, short bias
        if close_price < l3_level and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot or below L3 (failed breakout)
            exit_long = close_price <= pivot_level or close_price < l3_level
        elif position == -1:
            # Exit short if price returns to pivot or above H3 (failed breakout)
            exit_short = close_price >= pivot_level or close_price > h3_level
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals