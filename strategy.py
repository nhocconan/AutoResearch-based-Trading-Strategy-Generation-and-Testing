#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R1/S1 breakouts with 1d EMA50 trend filter and volume confirmation.
# Enter long when price breaks above 4h Camarilla R1 level with volume > 2.0x average and close > 1d EMA50 (bullish bias).
# Enter short when price breaks below 4h Camarilla S1 level with volume > 2.0x average and close < 1d EMA50 (bearish bias).
# Exit when price returns to the 4h Camarilla midpoint (P) or touches opposite level (S1 for long exit, R1 for short exit).
# Uses Camarilla structure for pivot points, higher timeframe EMA for trend filter, and volume for confirmation.
# Uses session filter (08-20 UTC) to avoid low-liquidity hours. Uses discrete position sizing (0.20) to control risk.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

name = "1h_Camarilla_R1S1_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead and TypeError
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivot calculation (HTF)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (using previous bar's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # True range for Camarilla calculation
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - close_4h)
    tr3 = np.abs(low_4h - close_4h)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Camarilla levels (based on previous bar's close and range)
    camarilla_pivot = close_4h  # Pivot is previous close
    camarilla_range = high_4h - low_4h
    
    # R1 and S1 levels (Camarilla: R1 = Close + Range * 1.1/12, S1 = Close - Range * 1.1/12)
    r1 = camarilla_pivot + camarilla_range * 1.1 / 12
    s1 = camarilla_pivot - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1d EMA50 bias
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > r1_aligned[i]
        short_breakout = close[i] < s1_aligned[i]
        
        # Exit conditions: return to pivot or opposite level touched
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and bullish_bias
        short_entry = short_breakout and vol_confirm and bearish_bias
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals