#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike
# Long when price breaks above H3 AND price > 4h EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below L3 AND price < 4h EMA50 AND volume > 2.0x 20-bar avg
# Exit when price crosses Camarilla H4/L4 levels (mean reversion to median)
# Uses discrete position sizing (0.20) to minimize fee drag.
# Target: 15-37 trades/year on 1h (60-150 total over 4 years).
# Camarilla levels provide high-probability reversal points; 4h EMA50 filters counter-trend moves.
# Volume spike ensures institutional participation, reducing false breakouts.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate EMA(50) on 4h data
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla levels (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Extract daily OHLC values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Align daily OHLC to 1h timeframe (each value represents the prior day's close)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Calculate Camarilla levels for each 1h bar based on prior day's OHLC
    # Camarilla H3/L3 and H4/L4 levels
    daily_range = daily_high_aligned - daily_low_aligned
    camarilla_h4 = daily_close_aligned + daily_range * 1.1 / 2
    camarilla_l4 = daily_close_aligned - daily_range * 1.1 / 2
    camarilla_h3 = daily_close_aligned + daily_range * 1.1 / 4
    camarilla_l3 = daily_close_aligned - daily_range * 1.1 / 4
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # EMA50 and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(camarilla_h4[i]) or 
            np.isnan(camarilla_l4[i]) or np.isnan(volume_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_4h_aligned[i]
        
        # Camarilla levels
        h3 = camarilla_h3[i]
        l3 = camarilla_l3[i]
        h4 = camarilla_h4[i]
        l4 = camarilla_l4[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below H4 (mean reversion to median)
            if curr_close < h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above L4 (mean reversion to median)
            if curr_close > l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long when price breaks above H3 AND price > 4h EMA50 AND volume confirmation
            if curr_close > h3 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below L3 AND price < 4h EMA50 AND volume confirmation
            elif curr_close < l3 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals