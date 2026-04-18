#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Reversal with Volume and Daily Trend Filter
# Camarilla pivot levels (R1, S1) often act as intraday support/resistance.
# In ranging markets, price tends to reverse at these levels.
# We use daily EMA34 to filter for the dominant trend direction to avoid counter-trend trades in strong moves.
# Volume confirmation ensures the reversal has participation.
# Designed for low trade frequency (15-35/year) to minimize fee drag in 6h timeframe.
# Works in ranging markets (reversals at S1/R1) and avoids losses in strong trends by following daily EMA.
name = "6h_Camarilla_R1S1_Reversal_Volume_DailyEMA34"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation and EMA filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # Formula: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # where C = (H+L+C)/3 (typical price)
    # We use previous day's data to avoid look-ahead
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    ph = df_1d['high'].iloc[-2:].values  # last two days, we'll use the first of these (yesterday)
    pl = df_1d['low'].iloc[-2:].values
    pc = df_1d['close'].iloc[-2:].values
    
    # For each day, calculate pivots based on the day before it
    # We need to align this to the 6h chart
    # Calculate pivots for each day (except the first day)
    pivot_high = []
    pivot_low = []
    pivot_close = []
    
    for i in range(1, len(df_1d)):
        ph_i = df_1d['high'].iloc[i-1]
        pl_i = df_1d['low'].iloc[i-1]
        pc_i = df_1d['close'].iloc[i-1]
        pivot_high.append(ph_i)
        pivot_low.append(pl_i)
        pivot_close.append(pc_i)
    
    # Convert to arrays and calculate pivot levels
    ph_arr = np.array(pivot_high)
    pl_arr = np.array(pivot_low)
    pc_arr = np.array(pivot_close)
    
    # Pivot point (not used directly in Camarilla but needed for calculations)
    # pp = (ph_arr + pl_arr + pc_arr) / 3
    
    # Calculate Camarilla levels
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    r1 = pc_arr + (ph_arr - pl_arr) * 1.1 / 12
    s1 = pc_arr - (ph_arr - pl_arr) * 1.1 / 12
    
    # We need to align these daily levels to the 6h timeframe
    # Each day's levels are valid for the entire next day
    r1_aligned = align_htf_to_ltf(prices, df_1d.iloc[1:], r1)  # skip first day as we don't have prior day
    s1_aligned = align_htf_to_ltf(prices, df_1d.iloc[1:], s1)
    
    # Calculate daily EMA34 for trend filter
    # Use close prices from df_1d
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 20-period average volume for confirmation (on 6h chart)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 00-24 UTC (we can trade all day for 6h timeframe, but avoid very low volume periods if needed)
    # For 6h, we'll use a simple session filter to avoid the quietest period (0-6 UTC)
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        # Avoid quietest period (0-6 UTC) for better volume
        in_session = not (0 <= hour < 6)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long reversal: price crosses below S1 AND volume confirmation AND price above daily EMA (uptrend)
            long_reversal = close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] if i > 0 else False
            if vol_confirm and long_reversal and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short reversal: price crosses above R1 AND volume confirmation AND price below daily EMA (downtrend)
            elif vol_confirm and close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] if i > 0 else False and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back above S1 (reversal failed) OR price crosses below daily EMA (trend change)
            exit_long = (close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1] if i > 0 else False) or close[i] < ema_34_aligned[i]
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back below R1 (reversal failed) OR price crosses above daily EMA (trend change)
            exit_short = (close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1] if i > 0 else False) or close[i] > ema_34_aligned[i]
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals