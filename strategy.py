#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme Reversal with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend filter and volume spike reference.
- Entry: Long when Williams %R(14) crosses above -80 (extreme oversold) AND price > 1d EMA34 (uptrend) AND 4h volume > 2.0 * 1d average volume per 4h bar;
         Short when Williams %R(14) crosses below -20 (extreme overbought) AND price < 1d EMA34 (downtrend) AND 4h volume > 2.0 * 1d average volume per 4h bar.
- Exit: Close-based reversal (opposite Williams %R extreme) or trend change (signal=0 when price crosses 1d EMA34 in opposite direction).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams %R captures exhaustion moves; 1d EMA34 ensures we trade with the dominant daily trend; volume spike confirms institutional participation.
- Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend) with trend filter to avoid counter-trend whipsaws.
- Estimated trades: ~100 total over 4 years (~25/year) based on Williams %R extreme frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and volume reference
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    avg_volume_1d = np.mean(volume_1d) if len(volume_1d) > 0 else 0
    expected_volume_4h = avg_volume_1d / 6  # 6x 4h bars per day
    
    # Calculate Williams %R(14) on 4h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, -100 * (highest_high - close) / denominator, -50)
    
    # Calculate Williams %R(14) on 1d data for HTF context (optional, not used in entry)
    highest_high_1d = pd.Series(df_1d['high'].values).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low_1d = pd.Series(df_1d['low'].values).rolling(window=lookback, min_periods=lookback).min().values
    denominator_1d = highest_high_1d - lowest_low_1d
    williams_r_1d = np.where(denominator_1d != 0, -100 * (highest_high_1d - close_1d) / denominator_1d, -50)
    
    # Align all indicators to primary 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    williams_r_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, lookback)  # Need sufficient data for EMA34 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Williams %R extreme levels
        williams_r_curr = williams_r_aligned[i]
        williams_r_prev = williams_r_aligned[i-1] if i > 0 else williams_r_curr
        
        # Cross above -80 (oversold recovery) or below -20 (overbought rejection)
        crossed_above_oversold = (williams_r_prev <= -80) and (williams_r_curr > -80)
        crossed_below_overbought = (williams_r_prev >= -20) and (williams_r_curr < -20)
        
        # Trend filter: price vs 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation: 4h volume > 2.0 * expected 4h volume from 1d average
        vol_spike = curr_volume > 2.0 * expected_volume_4h
        
        # Exit conditions
        if position != 0:
            # Exit long: price crosses below 1d EMA34 (trend breakdown) or Williams %R crosses above -20 (overbought)
            if position == 1:
                if curr_close <= ema_34_1d_aligned[i] or crossed_below_overbought:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses above 1d EMA34 (trend reversal) or Williams %R crosses below -80 (oversold)
            elif position == -1:
                if curr_close >= ema_34_1d_aligned[i] or crossed_above_oversold:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions
        if position == 0:
            # Check for entry signals
            if vol_spike:
                # Long: Williams %R crosses above -80 (oversold recovery) AND uptrend
                if crossed_above_oversold and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 (overbought rejection) AND downtrend
                elif crossed_below_overbought and downtrend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_Reversal_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0