#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1w trend filter and volume spike confirmation.
# Uses 6h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years).
# 1w EMA34 provides primary trend filter: bull when price > EMA34, bear when price < EMA34.
# 6h Camarilla R3/S3 levels provide breakout signals with proven edge from prior experiments.
# Volume spike (>2.0x 20-bar average) confirms breakout strength.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn.
# Works in both bull and bear markets via trend filter + breakout logic.

name = "6h_Camarilla_R3_S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Camarilla levels and 1w data for EMA34 trend
    df_6h = get_htf_data(prices, '6h')
    df_1w = get_htf_data(prices, '1w')
    if len(df_6h) < 20 or len(df_1w) < 34:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 6h Camarilla levels (based on previous day's range)
    # Camarilla: H5 = close + 1.1*(high-low)*1.1/2, L5 = close - 1.1*(high-low)*1.1/2
    # But standard Camarilla uses previous period's high/low
    # For intraday, we use previous 6h bar's range
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close = np.roll(close_6h, 1) if 'close_6h' in locals() else df_6h['close'].values
    prev_close = np.roll(df_6h['close'].values, 1)
    
    # Handle first bar
    prev_high[0] = high_6h[0]
    prev_low[0] = low_6h[0]
    prev_close[0] = close_6h[0]
    
    range_6h = prev_high - prev_low
    camarilla_h3 = prev_close + 1.1 * range_6h * 1.1 / 4  # H3 level
    camarilla_l3 = prev_close - 1.1 * range_6h * 1.1 / 4  # L3 level
    camarilla_h4 = prev_close + 1.1 * range_6h * 1.1 / 2  # H4 level (breakout)
    camarilla_l4 = prev_close - 1.1 * range_6h * 1.1 / 2  # L4 level (breakout)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_l4)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient history for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA34 direction (price above/below EMA34)
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        # Camarilla breakout conditions (using H4/L4 for breakout)
        long_breakout = close[i] > camarilla_h4_aligned[i]
        short_breakout = close[i] < camarilla_l4_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = price_above_ema and long_breakout and vol_confirm
        short_entry = price_below_ema and short_breakout and vol_confirm
        
        # Exit conditions: opposite H3/L3 level (mean reversion to mean)
        long_exit = close[i] < camarilla_l3_aligned[i]  # Exit long at L3
        short_exit = close[i] > camarilla_h3_aligned[i]  # Exit short at H3
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals