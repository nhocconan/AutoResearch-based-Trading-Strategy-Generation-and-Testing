#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot long/short with 1d trend filter and volume confirmation
# - Long: price touches Camarilla L3 (support) from below, 1d EMA(50) uptrend, volume > 1.3x 20-period avg
# - Short: price touches Camarilla H3 (resistance) from above, 1d EMA(50) downtrend, volume > 1.3x 20-period avg
# - Exit: price reaches Camarilla L4/H4 (extreme levels) or opposite H3/L3 level
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 20-35 trades/year (80-140 total over 4 years) to stay within fee drag limits
# - Camarilla levels provide institutional support/resistance that work in ranging and trending markets

name = "4h_1d_camarilla_pivot_volume_v1"
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
    
    # Load 1d data ONCE before loop for trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 4h Camarilla levels from previous day's OHLC
    # Need previous day's high, low, close - using 1d data shifted by 1
    if len(df_1d) < 2:
        return signals
        
    # Previous day's OHLC (1d data)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla formulas:
    # H4 = close + ((high - low) * 1.1 / 2)
    # H3 = close + ((high - low) * 1.1 / 4)
    # L3 = close - ((high - low) * 1.1 / 4)
    # L4 = close - ((high - low) * 1.1 / 2)
    camarilla_h4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_h3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_l3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    camarilla_l4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # 1d EMA trend filter
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: price touches L3 from below with volume confirmation and long-term uptrend
        if close_price <= l3 and low[i] < l3 and close_price > l3 * 0.999:  # Touched L3 from below
            if vol_confirm and ema_bias_long:
                enter_long = True
        
        # Short: price touches H3 from above with volume confirmation and long-term downtrend
        if close_price >= h3 and high[i] > h3 and close_price < h3 * 1.001:  # Touched H3 from above
            if vol_confirm and ema_bias_short:
                enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reaches L4 (extreme support) or crosses above H3 (opposite resistance)
            exit_long = close_price <= l4 or close_price >= h3
        elif position == -1:
            # Exit short if price reaches H4 (extreme resistance) or crosses below L3 (opposite support)
            exit_short = close_price >= h4 or close_price <= l3
        
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