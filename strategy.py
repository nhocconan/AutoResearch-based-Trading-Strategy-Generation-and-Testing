#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla H4/L4 levels with 1d EMA34 trend filter and volume confirmation.
# Enter long when price breaks above 1w Camarilla H4 level with volume > 1.5x average and close > 1d EMA34 (bullish bias).
# Enter short when price breaks below 1w Camarilla L4 level with volume > 1.5x average and close < 1d EMA34 (bearish bias).
# Exit when price returns to the 1w Camarilla midpoint (P5) or touches the opposite level (L4 for long exit, H4 for short exit).
# Uses discrete position sizing (0.25) to control risk and minimize fee churn. Target: 30-100 total trades over 4 years.
# Works in bull markets (breakouts continue up with trend) and bear markets (breakdowns continue down with trend).

name = "1d_Camarilla_H4L4_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivot calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w Camarilla levels (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True range for Camarilla calculation
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - close_1w)
    tr3 = np.abs(low_1w - close_1w)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Camarilla levels (based on previous week's close and range)
    camarilla_pivot = close_1w  # Pivot is previous close
    camarilla_range = high_1w - low_1w
    
    # H4 and L4 levels (stronger breakout levels for fewer trades)
    h4 = camarilla_pivot + camarilla_range * 1.1 / 2
    l4 = camarilla_pivot - camarilla_range * 1.1 / 2
    p5 = camarilla_pivot  # Midpoint for exit (same as pivot in Camarilla)
    
    # Align Camarilla levels to 1d timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    p5_aligned = align_htf_to_ltf(prices, df_1w, p5)
    
    # Get 1d data for EMA34 trend filter (same timeframe, but we still use aligned for consistency)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(p5_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1d EMA34 bias
        bullish_bias = close[i] > ema_34_1d_aligned[i]
        bearish_bias = close[i] < ema_34_1d_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > h4_aligned[i]
        short_breakout = close[i] < l4_aligned[i]
        
        # Exit conditions: return to pivot (P5) or touch opposite level
        long_exit = close[i] < p5_aligned[i]
        short_exit = close[i] > p5_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and bullish_bias
        short_entry = short_breakout and vol_confirm and bearish_bias
        
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