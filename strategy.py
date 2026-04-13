#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot bounce with volume confirmation and daily trend filter.
# Camarilla levels provide strong support/resistance in ranging markets.
# Volume confirmation ensures bounces have institutional participation.
# Daily trend filter aligns with higher timeframe direction.
# Target: 20-40 trades per year (80-160 total over 4 years) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels (using previous day's OHLC)
    # For each bar, we use the previous completed day's data
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    prev_day_close = np.full(n, np.nan)
    
    day_index = 0
    for i in range(n):
        current_time = pd.Timestamp(prices['open_time'].iloc[i])
        current_date = current_time.date()
        
        # Advance day_index to match current date
        while day_index < len(df_1d) and pd.Timestamp(df_1d.index[day_index]).date() < current_date:
            day_index += 1
        
        # Use previous day's data (day_index - 1) if available
        if day_index > 0:
            prev_day = day_index - 1
            prev_day_high[i] = df_1d['high'].iloc[prev_day]
            prev_day_low[i] = df_1d['low'].iloc[prev_day]
            prev_day_close[i] = df_1d['close'].iloc[prev_day]
    
    # Calculate Camarilla levels from previous day's data
    camarilla_h5 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_l5 = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(prev_day_high[i]) or np.isnan(prev_day_low[i]) or np.isnan(prev_day_close[i])):
            range_val = prev_day_high[i] - prev_day_low[i]
            camarilla_h5[i] = prev_day_close[i] + range_val * 1.1 / 2
            camarilla_h4[i] = prev_day_close[i] + range_val * 1.1 / 4
            camarilla_h3[i] = prev_day_close[i] + range_val * 1.1 / 6
            camarilla_l3[i] = prev_day_close[i] - range_val * 1.1 / 6
            camarilla_l4[i] = prev_day_close[i] - range_val * 1.1 / 4
            camarilla_l5[i] = prev_day_close[i] - range_val * 1.1 / 2
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate daily EMA trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (21 + 1)
    ema_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema_1d[i] = (close_1d[i] - ema_1d[i-1]) * ema_multiplier + ema_1d[i-1]
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        daily_ema = ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: price bounces off Camarilla L3 with volume + above daily EMA
            if (price <= camarilla_l3[i] * 1.005 and  # Allow small tolerance
                price >= camarilla_l3[i] * 0.995 and
                volume_confirm and
                price > daily_ema):
                position = 1
                signals[i] = position_size
            # Short: price bounces off Camarilla H3 with volume + below daily EMA
            elif (price >= camarilla_h3[i] * 0.995 and
                  price <= camarilla_h3[i] * 1.005 and
                  volume_confirm and
                  price < daily_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches Camarilla H3 or H4
            if (price >= camarilla_h3[i] * 0.995 or
                price >= camarilla_h4[i] * 0.995):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches Camarilla L3 or L4
            if (price <= camarilla_l3[i] * 1.005 or
                price <= camarilla_l4[i] * 1.005):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Bounce_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0