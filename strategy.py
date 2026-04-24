#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme reversal with 1w EMA200 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 80-120 total trades over 4 years (20-30/year).
- HTF: 1w for EMA200 trend filter and 1d for Williams %R calculation.
- Entry: Long when Williams %R(14) crosses above -80 (oversold reversal) AND price > 1w EMA200 AND volume > 1.5x 20-period MA.
         Short when Williams %R(14) crosses below -20 (overbought reversal) AND price < 1w EMA200 AND volume > 1.5x 20-period MA.
- Exit: Opposite Williams %R extreme OR price crosses 1w EMA200 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R identifies momentum exhaustion; EMA200 filter ensures trades align with weekly trend.
- Volume confirmation avoids low-conviction reversals.
- Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on reversal frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def williams_r(high, low, close, period):
    """Calculate Williams %R."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 210:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    ema200_1w = ema(df_1w['close'].values, 200)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w, additional_delay_bars=1)
    
    # Calculate 1d Williams %R(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    wr_14 = williams_r(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    wr_14_aligned = align_htf_to_ltf(prices, df_1d, wr_14, additional_delay_bars=1)
    
    # Calculate 1d volume SMA(20) for confirmation
    vol_sma_20 = sma(df_1d['volume'].values, 20)
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 210  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(wr_14_aligned[i]) or 
            np.isnan(vol_sma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Williams %R extreme OR price crosses 1w EMA200 in opposite direction
        if position != 0:
            # Exit long: Williams %R rises above -20 (overbought) OR price falls below 1w EMA200
            if position == 1:
                if wr_14_aligned[i] > -20 or curr_close < ema200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R falls below -80 (oversold) OR price rises above 1w EMA200
            elif position == -1:
                if wr_14_aligned[i] < -80 or curr_close > ema200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme reversal with trend and volume filters
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND bullish weekly trend AND volume confirmation
            if (wr_14_aligned[i] > -80 and 
                i > start_idx and wr_14_aligned[i-1] <= -80 and  # Crossed above -80
                curr_close > ema200_1w_aligned[i] and 
                curr_volume > 1.5 * vol_sma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND bearish weekly trend AND volume confirmation
            elif (wr_14_aligned[i] < -20 and 
                  i > start_idx and wr_14_aligned[i-1] >= -20 and  # Crossed below -20
                  curr_close < ema200_1w_aligned[i] and 
                  curr_volume > 1.5 * vol_sma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_ExtremeReversal_1wEMA200_TrendFilter_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0