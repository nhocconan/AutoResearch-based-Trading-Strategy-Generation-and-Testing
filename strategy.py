#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend filter and Williams %R calculation.
- Entry: Long when Williams %R(14) crosses above -80 from below AND price > 1d EMA50 AND volume > 2.0 * 6h volume MA(20);
         Short when Williams %R(14) crosses below -20 from above AND price < 1d EMA50 AND volume > 2.0 * 6h volume MA(20).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when price crosses 1d EMA50).
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R identifies overextended moves; EMA50 ensures we trade with the daily trend; volume confirmation validates reversal strength.
- Works in bull markets (buy oversold bounces in uptrend) and bear markets (sell overbought bounces in downtrend) with daily trend filter to avoid counter-trend whipsaws.
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
    
    # Get 1d data for EMA50 trend filter and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R(14) on 1d
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 6h data for volume MA
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for Williams %R(14) and EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit: trend change (price crosses 1d EMA50)
        if position != 0:
            if position == 1 and curr_close < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and curr_close > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Williams %R extremes and crossover detection
        wr_prev = williams_r_aligned[i-1] if i > 0 else williams_r_aligned[i]
        wr_curr = williams_r_aligned[i]
        
        # Bullish: Williams %R crosses above -80 from below (oversold bounce)
        bullish_cross = (wr_prev <= -80) and (wr_curr > -80)
        # Bearish: Williams %R crosses below -20 from above (overbought bounce)
        bearish_cross = (wr_prev >= -20) and (wr_curr < -20)
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma_6h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Williams %R bullish crossover AND uptrend (price > EMA50)
                if bullish_cross and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R bearish crossover AND downtrend (price < EMA50)
                elif bearish_cross and downtrend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_Reversal_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0