#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above R1 AND price > 4h EMA50 AND volume > 1.8x 24-bar avg
# Short when price breaks below S1 AND price < 4h EMA50 AND volume > 1.8x 24-bar avg
# Exit when price crosses opposite Camarilla level (S1 for longs, R1 for shorts)
# Uses discrete position sizing (0.20) to minimize fee churn.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h.
# Camarilla levels provide mathematical support/resistance; 4h EMA50 filters counter-trend moves.
# Volume spike ensures institutional participation, reducing false breakouts.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within trend via exits).

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeConfirm_v1"
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
    
    # Precompute session hours (08-20 UTC)
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
    
    # Get 1d data for Camarilla pivot levels (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Extract prior day's OHLC (1d timeframe)
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    # Set first value to NaN as we don't have prior day
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Align prior day OHLC to 1h timeframe
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
    
    # Calculate Camarilla levels for each 1h bar based on prior day's OHLC
    # Camarilla R1 = Close + (High - Low) * 1.1/12
    # Camarilla S1 = Close - (High - Low) * 1.1/12
    # We use R1/S1 for entries/exits as they provide tighter entries
    range_hl = prior_high_aligned - prior_low_aligned
    r1 = prior_close_aligned + range_hl * 1.1 / 12
    s1 = prior_close_aligned - range_hl * 1.1 / 12
    
    # Volume confirmation: >1.8x 24-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.8 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1) + 1  # EMA50 warmup + 1 for prior day shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(volume_ma_24[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_4h_aligned[i]
        
        # Camarilla levels
        r1_level = r1[i]
        s1_level = s1[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below S1 (mean reversion to median)
            if curr_close < s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above R1 (mean reversion to median)
            if curr_close > r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long when price breaks above R1 AND price > 4h EMA50 AND volume confirmation
            if curr_close > r1_level and curr_close > ema_50 and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S1 AND price < 4h EMA50 AND volume confirmation
            elif curr_close < s1_level and curr_close < ema_50 and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals