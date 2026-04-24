#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA50 Trend Filter and Volume Spike Confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend filter and volume MA reference.
- Entry: Long when Williams %R(14) < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 2.0 * 6h volume MA(20);
         Short when Williams %R(14) > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 2.0 * 6h volume MA(20).
- Exit: Close-based reversal (opposite Williams %R extreme) or trend change (signal=0 when price crosses 1d EMA50 opposite direction).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams %R identifies exhaustion points in both bull and bear markets; 1d EMA50 ensures we trade with the higher timeframe trend;
  volume spike confirms institutional participation at reversal points. Works in ranging markets (mean reversion at extremes)
  and trending markets (pullbacks to EMA50 in direction of trend).
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
    
    # Get 6h data for volume MA reference and Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low + 1e-10) * -100
    # Avoid division by zero when high == low
    
    # Get 6h volume MA for confirmation
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 6h timeframe (though primary is 6h, we still align for safety)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA50 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit: trend change (price crosses 1d EMA50 in opposite direction of position)
        if position != 0:
            if position == 1 and curr_close < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and curr_close > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation and Williams %R extremes
        vol_confirm = curr_volume > 2.0 * vol_ma_6h_aligned[i]
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Oversold AND uptrend (price above EMA50)
                if oversold and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Overbought AND downtrend (price below EMA50)
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

name = "6h_WilliamsR_Extreme_Reversal_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0