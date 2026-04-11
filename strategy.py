#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and 1d trend filter
# - Long: price breaks above Camarilla H3 level, volume > 1.5x 20-period avg, price > 1d EMA(50)
# - Short: price breaks below Camarilla L3 level, volume > 1.5x 20-period avg, price < 1d EMA(50)
# - Exit: price returns to Camarilla pivot point (PP) or opposite Camarilla level (L3 for long, H3 for short)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Camarilla levels provide structured support/resistance that work in both trending and ranging markets

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(volume_sma_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Calculate 12h Camarilla levels using previous bar's OHLC
        if i == 0:
            # First bar - use current bar (will be corrected on next iteration)
            ph, pl, pc = high_price, low_price, close_price
        else:
            ph = high[i-1]  # Previous high
            pl = low[i-1]   # Previous low
            pc = close[i-1] # Previous close
        
        # Camarilla levels calculation
        range_val = ph - pl
        if range_val <= 0:
            # Avoid division by zero or negative range
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
            
        # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
        # We use H3 and L3 for breakouts, PP for exit
        camarilla_pp = pc + (range_val * 1.1) / 2
        camarilla_h3 = pc + (range_val * 1.1) / 4
        camarilla_l3 = pc - (range_val * 1.1) / 4
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1d EMA trend bias
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above Camarilla H3, volume confirmation, long bias
        if close_price > camarilla_h3 and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short breakout: price closes below Camarilla L3, volume confirmation, short bias
        if close_price < camarilla_l3 and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to or below Camarilla pivot point
            exit_long = close_price <= camarilla_pp
        elif position == -1:
            # Exit short if price returns to or above Camarilla pivot point
            exit_short = close_price >= camarilla_pp
        
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