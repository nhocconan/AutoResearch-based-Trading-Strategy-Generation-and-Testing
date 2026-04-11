#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and 1d EMA trend filter
# - Long: price breaks above Camarilla H3 level, volume > 1.3x 20-period avg, price > 1d EMA(50)
# - Short: price breaks below Camarilla L3 level, volume > 1.3x 20-period avg, price < 1d EMA(50)
# - Exit: price returns to Camarilla pivot point (midpoint)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 20-35 trades/year (80-140 total over 4 years) to stay within fee drag limits
# - Camarilla levels provide intraday support/resistance that work in both trending and ranging markets

name = "4h_1d_camarilla_breakout_v1"
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
    
    # Load 1d data ONCE before loop for EMA trend filter and Camarilla calculation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: H4 = C + ((H-L)*1.1/2), H3 = C + ((H-L)*1.1/4), L3 = C - ((H-L)*1.1/4), L4 = C - ((H-L)*1.1/2)
    # Where C = close, H = high, L = low of previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_range = prev_high - prev_low
    camarilla_h3 = camarilla_pivot + (camarilla_range * 1.1 / 4)
    camarilla_l3 = camarilla_pivot - (camarilla_range * 1.1 / 4)
    camarilla_h4 = camarilla_pivot + (camarilla_range * 1.1 / 2)
    camarilla_l4 = camarilla_pivot - (camarilla_range * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        pivot_level = camarilla_pivot_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # 1d EMA trend filter
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Camarilla H3, volume confirmation, long bias
        if close_price > h3_level and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below Camarilla L3, volume confirmation, short bias
        if close_price < l3_level and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to Camarilla pivot point
            exit_long = close_price <= pivot_level
        elif position == -1:
            # Exit short if price returns to Camarilla pivot point
            exit_short = close_price >= pivot_level
        
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