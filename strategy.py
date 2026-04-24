#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme reversal with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend filter.
- Entry: Long when Williams %R < -80 (oversold) AND price > EMA50_1d (uptrend) AND volume > 2.0 * 6h volume MA(20);
         Short when Williams %R > -20 (overbought) AND price < EMA50_1d (downtrend) AND volume > 2.0 * 6h volume MA(20).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when price crosses EMA50_1d).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams %R identifies exhaustion points; EMA50 trend filter ensures we trade with the daily trend; volume confirmation avoids false reversals.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R (14-period) on 6h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Get 6h data for volume MA
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)  # Williams %R needs no extra delay
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_6h)  # approximate alignment for volume MA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA50 and Williams %R(14)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit: trend change (price crosses EMA50_1d)
        if position != 0:
            if position == 1 and curr_close <= ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and curr_close >= ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation and Williams %R extremes
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Trend filter: only trade in direction of price vs EMA50_1d
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 2.0 * vol_ma_6h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Williams %R oversold AND uptrend
                if oversold and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R overbought AND downtrend
                elif overbought and downtrend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_EMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0