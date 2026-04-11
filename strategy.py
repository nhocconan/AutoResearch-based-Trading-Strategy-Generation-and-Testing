#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and 12h trend filter
# - Long: price breaks above Camarilla H3 level, volume > 1.3x 20-period avg, 12h EMA(20) rising
# - Short: price breaks below Camarilla L3 level, volume > 1.3x 20-period avg, 12h EMA(20) falling
# - Exit: price returns to Camarilla pivot point (H3/L3 for stop, pivot for target)
# - Uses 12h EMA(20) for trend filter and 1d for pivot calculation (more stable than intraday)
# - Target: 25-35 trades/year (100-140 total over 4 years) to stay within fee drag limits
# - Camarilla levels work well in ranging markets; breakouts with volume work in trending markets
# - Volume confirmation reduces false breakouts; 12h trend ensures alignment with higher timeframe

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
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
    # L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    # Pivot = (high + low + close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Load 12h data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return signals
    
    # Pre-compute 12h EMA(20) for trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(ema_20_12h_aligned[i])):
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
        
        # 12h EMA trend: rising if current > previous, falling if current < previous
        # Need to check previous bar's EMA to determine slope
        if i > 100:
            ema_prev = ema_20_12h_aligned[i-1]
            ema_curr = ema_20_12h_aligned[i]
            ema_rising = ema_curr > ema_prev
            ema_falling = ema_curr < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above H3, volume confirmation, 12h EMA rising
        if close_price > h3_level and vol_confirm and ema_rising:
            enter_long = True
        
        # Short breakout: price below L3, volume confirmation, 12h EMA falling
        if close_price < l3_level and vol_confirm and ema_falling:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot level or breaks below L3 (stop)
            exit_long = close_price <= pivot_level
        elif position == -1:
            # Exit short if price returns to pivot level or breaks above H3 (stop)
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