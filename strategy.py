#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Supertrend (ATR=10, mult=3) + 1d volume spike + price > 1d EMA200 trend filter.
# Long: Supertrend green (uptrend) + volume > 2x 20-period average volume + close > EMA200.
# Short: Supertrend red (downtrend) + volume > 2x average volume + close < EMA200.
# Uses 1d timeframe for trend filter and volume confirmation, 12h for Supertrend signal.
# Works in bull/bear by requiring volume spikes and trend alignment.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA200 and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 1d average volume (20-period)
    avg_volume_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        avg_volume_1d[i] = np.mean(volume_1d[i-20:i])
    
    # 12h data for Supertrend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Supertrend (ATR=10, mult=3)
    def atr(high, low, close, period):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first TR is just high-low
        atr_vals = np.zeros_like(close)
        atr_vals[:period] = np.nan
        for i in range(period, len(tr)):
            atr_vals[i] = (atr_vals[i-1] * (period-1) + tr[i]) / period
        return atr_vals
    
    atr_vals = atr(high_12h, low_12h, close_12h, 10)
    
    # Basic upper and lower bands
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + 3 * atr_vals
    lower_band = hl2 - 3 * atr_vals
    
    # Supertrend calculation
    supertrend = np.full_like(close_12h, np.nan)
    direction = np.full_like(close_12h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(10, len(close_12h)):
        if np.isnan(atr_vals[i-1]) or np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
            continue
            
        if i == 10:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            if close_12h[i-1] > supertrend[i-1]:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                direction[i] = 1
            else:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                direction[i] = -1
    
    # Align 1d indicators to 12h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(df_12h, df_1d, ema_200_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(df_12h, df_1d, avg_volume_1d)
    
    # Align Supertrend to 12h (already in 12h, but need to align to 12h index for consistency)
    # Actually, Supertrend is already calculated on 12h data, so we just need to align it to 12h index
    # But since we're using 12h data directly, we can use it as is for 12h timeframe
    # However, we need to align it to the 12h dataframe's index for consistency with other aligned arrays
    # For simplicity, we'll use the Supertrend and direction arrays directly since they're on 12h
    
    # Now align 12h data to the main timeframe (which is also 12h in this case)
    # Since main timeframe is 12h, we don't need alignment - but we'll use the same approach for consistency
    # We need to get the main timeframe index aligned with 12h data
    # Actually, since we're using 12h data for signals and main timeframe is 12h, we can proceed
    
    # But wait - the main timeframe is 12h, so prices dataframe is already at 12h resolution
    # Therefore, we can use the 12h indicators directly
    
    # However, to be consistent with the MTF approach and avoid confusion, let's align properly
    # We'll treat the 12h data as our HTF and align to the main timeframe (which is also 12h)
    # This should give us a 1:1 alignment
    
    # Re-align Supertrend and direction to ensure proper indexing
    supertrend_aligned = align_htf_to_ltf(df_12h, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(df_12h, df_12h, direction)
    
    # Align 1d indicators to main timeframe (12h)
    ema_200_1d_aligned = align_htf_to_ltf(df_12h, df_1d, ema_200_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(df_12h, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(10, n):
        # Skip if any required data is not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume_1d_aligned[i]
        st_direction = direction_aligned[i]
        ema_trend = ema_200_1d_aligned[i]
        
        # Volume confirmation: current volume > 2x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: uptrend + above EMA200 + volume confirmation
            if (st_direction == 1 and 
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: downtrend + below EMA200 + volume confirmation
            elif (st_direction == -1 and 
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: downtrend or price below EMA200
            if (st_direction == -1 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: uptrend or price above EMA200
            if (st_direction == 1 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Supertrend_EMA200_Volume"
timeframe = "12h"
leverage = 1.0