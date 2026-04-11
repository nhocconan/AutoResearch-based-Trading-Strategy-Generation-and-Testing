#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and 12h trend filter
# - Long: price breaks above Camarilla H3 level, volume > 1.3x 20-period avg, 12h EMA(20) > EMA(50)
# - Short: price breaks below Camarilla L3 level, volume > 1.3x 20-period avg, 12h EMA(20) < EMA(50)
# - Exit: price returns to Camarilla pivot point (PP)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 20-30 trades/year (80-120 total over 4 years) to stay within fee drag limits
# - Camarilla levels derived from prior day's range, effective in both trending and ranging markets

name = "4h_12h_camarilla_breakout_v1"
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
    
    # Load 12h data ONCE before loop for trend filter (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Pre-compute 12h EMAs for trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 1d high/low for Camarilla levels (using prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d bar
    # H4 = Close + 1.5*(High-Low), H3 = Close + 1.1*(High-Low), etc.
    camarilla_pp = (high_1d + low_1d + close_1d) / 3
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (using prior completed 1d bar)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(volume_sma_20[i]) or
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels from prior 1d bar
        pp = camarilla_pp_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # 12h trend filter: EMA(20) > EMA(50) for long bias, < for short bias
        ema_bias_long = ema_20_12h_aligned[i] > ema_50_12h_aligned[i]
        ema_bias_short = ema_20_12h_aligned[i] < ema_50_12h_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above H3, volume confirmation, long trend bias
        if close_price > h3 and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below L3, volume confirmation, short trend bias
        if close_price < l3 and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point
            exit_long = close_price <= pp
        elif position == -1:
            # Exit short if price returns to pivot point
            exit_short = close_price >= pp
        
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