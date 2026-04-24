#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA50 trend direction (long only above EMA50, short only below).
- Camarilla levels: H3/L3 from prior 4h bar for breakout entries.
- Volume filter: 1h volume > 1.5 * 20-period average volume.
- Session filter: Trade only 08:00-20:00 UTC to avoid low-liquidity hours.
- Signal size: 0.20 discrete to minimize fee drag.
- Works in bull/bear by only taking breakouts in direction of 4h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08:00-20:00 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # --- 4h HTF: EMA50 for trend direction ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # --- Prior 4h bar OHLC for Camarilla calculation ---
    # We need the completed 4h bar before current 1h bar
    df_4h_complete = get_htf_data(prices, '4h')  # Same data, we'll use its completed bars
    high_4h = df_4h_complete['high'].values
    low_4h = df_4h_complete['low'].values
    close_4h = df_4h_complete['close'].values
    
    # Camarilla levels: H3/L3 = close +- (high-low)*1.1/4
    # These are based on the COMPLETED 4h bar, so we shift by 1 to avoid look-ahead
    # align_htf_to_ltf with additional_delay_bars=1 ensures we use previous completed 4h bar
    hl_range_4h = high_4h - low_4h
    camarilla_h3_4h = close_4h + hl_range_4h * 1.1 / 4
    camarilla_l3_4h = close_4h - hl_range_4h * 1.1 / 4
    
    # Align to 1h with 1-bar delay (use previous completed 4h bar)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h_complete, camarilla_h3_4h, additional_delay_bars=1)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h_complete, camarilla_l3_4h, additional_delay_bars=1)
    
    # --- 1h volume confirmation: >1.5 * 20-period average ---
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(h3_4h_aligned[i]) or 
            np.isnan(l3_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: EMA50 direction from 4h
        uptrend = ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] if i > 0 else False  # Rising EMA = uptrend
        downtrend = ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] if i > 0 else False  # Falling EMA = downtrend
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Camarilla breakout levels from prior 4h bar
        h3 = h3_4h_aligned[i]
        l3 = l3_4h_aligned[i]
        
        # Exit conditions: reverse Camarilla breakout or trend change
        if position != 0:
            # Exit long: price < L3 or trend turns down
            if position == 1:
                if curr_close < l3 or downtrend:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > H3 or trend turns up
            elif position == -1:
                if curr_close > h3 or uptrend:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume filters
        if position == 0:
            # Long: price > H3 AND uptrend AND volume confirmation
            long_condition = (curr_close > h3 and 
                            uptrend and
                            volume_confirm)
            
            # Short: price < L3 AND downtrend AND volume confirmation
            short_condition = (curr_close < l3 and 
                             downtrend and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_CamarillaH3L3_Breakout_4hEMA50Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0