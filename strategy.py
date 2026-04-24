#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme reversal with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter.
- Entry: Long when Williams %R(14) crosses above -80 from below AND price > 1d EMA34 (uptrend) AND volume > 2.0 * 6h volume MA(20);
         Short when Williams %R(14) crosses below -20 from above AND price < 1d EMA34 (downtrend) AND volume > 2.0 * 6h volume MA(20).
- Exit: Close-based reversal (opposite signal) or when Williams %R returns to neutral zone (-80 to -20).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams %R identifies overbought/oversold conditions; EMA34 trend filter ensures we trade with the daily trend to avoid counter-trend whipsaws; volume spike confirmation (2.0x) avoids false reversals.
- Works in bull markets (buy oversold bounces in uptrend) and bear markets (sell overbought reversals in downtrend).
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
    
    # Calculate Williams %R(14) on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # avoid division by zero
    
    # Calculate Williams %R(14) slope for crossover detection
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = williams_r[0]
    williams_r_cross_up = (williams_r > -80) & (williams_r_prev <= -80)  # crosses above -80
    williams_r_cross_down = (williams_r < -20) & (williams_r_prev >= -20)  # crosses below -20
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 6h data for volume MA
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, prices, vol_ma_6h)  # same timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 34) + 5  # Williams %R needs 14, EMA34 needs 34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_6h_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(williams_r_cross_up[i]) or np.isnan(williams_r_cross_down[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit: Williams %R returns to neutral zone (-80 to -20) or opposite crossover
        if position != 0:
            if position == 1 and (williams_r[i] >= -20 or williams_r_cross_down[i]):
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and (williams_r[i] <= -80 or williams_r_cross_up[i]):
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume spike confirmation and trend filter
        vol_confirm = curr_volume > 2.0 * vol_ma_6h_aligned[i]
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Williams %R crosses above -80 from below AND uptrend
                if williams_r_cross_up[i] and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 from above AND downtrend
                elif williams_r_cross_down[i] and downtrend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_Reversal_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0