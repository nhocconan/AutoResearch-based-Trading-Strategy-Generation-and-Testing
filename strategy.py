#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend filter.
- Entry: Long when Williams %R(14) crosses above -80 (oversold) AND 1d EMA50 > 1d EMA50(previous) (uptrend) AND volume > 2.0 * 4h volume MA(20);
         Short when Williams %R(14) crosses below -20 (overbought) AND 1d EMA50 < 1d EMA50(previous) (downtrend) AND volume > 2.0 * 4h volume MA(20).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when EMA50 slope changes sign).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams %R identifies overextended moves; EMA50 trend filter ensures we trade with the higher timeframe trend; volume spike confirms institutional participation.
- Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend) with trend filter to avoid counter-trend whipsaws.
- Estimated trades: ~100 total over 4 years (~25/year) based on Williams %R extreme readings with filters.
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
    ema_50_slope = ema_50_1d - np.roll(ema_50_1d, 1)
    ema_50_slope[0] = 0
    
    # Calculate Williams %R(14) on 4h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # Avoid division by zero
    
    # Get 4h data for volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_50_slope)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA50 and Williams %R(14)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_slope_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit: trend change (EMA50 slope changes sign)
        if position != 0:
            if position == 1 and ema_50_slope_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and ema_50_slope_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
                continue
        
        # Williams %R signals
        wr_prev = williams_r_aligned[i-1] if i > 0 else williams_r_aligned[i]
        wr_curr = williams_r_aligned[i]
        
        # Oversold condition: Williams %R crosses above -80
        oversold_signal = wr_prev <= -80 and wr_curr > -80
        # Overbought condition: Williams %R crosses below -20
        overbought_signal = wr_prev >= -20 and wr_curr < -20
        
        # Trend filter: only trade in direction of 1d EMA50 slope
        uptrend = ema_50_slope_aligned[i] > 0
        downtrend = ema_50_slope_aligned[i] < 0
        
        # Volume confirmation (require strong volume spike)
        vol_confirm = curr_volume > 2.0 * vol_ma_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Williams %R crosses above -80 AND uptrend
                if oversold_signal and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 AND downtrend
                elif overbought_signal and downtrend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_MeanReversion_EMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0