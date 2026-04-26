#!/usr/bin/env python3
"""
6h_RSI_Trend_MeanReversion_Combo_v1
Hypothesis: On 6h timeframe, combine RSI mean reversion in ranging markets (RSI < 30 for long, > 70 for short) with trend following in trending markets (price > EMA50 for long, < EMA50 for short) using ADX regime filter. ADX > 25 triggers trend mode, ADX < 20 triggers mean reversion mode. Uses 1d HTF for EMA50 and ADX calculation to avoid look-ahead. Discrete sizing (0.0, ±0.25) minimizes fee churn. Targets 50-150 trades over 4 years (12-37/year) for optimal 6h frequency. Works in bull markets via trend following and bear markets via mean reversion of overextended moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for HTF indicators (EMA50 and ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need enough for EMA50 and ADX
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily ADX (14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d_arr[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_arr[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+
    tr_period = 14
    tr_sum = np.zeros_like(tr)
    dm_plus_sum = np.zeros_like(dm_plus)
    dm_minus_sum = np.zeros_like(dm_minus)
    
    # Initial values (simple average)
    if len(tr) >= tr_period:
        tr_sum[tr_period-1] = np.nansum(tr[1:tr_period])  # skip index 0 NaN
        dm_plus_sum[tr_period-1] = np.nansum(dm_plus[1:tr_period])
        dm_minus_sum[tr_period-1] = np.nansum(dm_minus[1:tr_period])
        
        # Wilder smoothing
        for i in range(tr_period, len(tr)):
            tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
            dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / tr_period) + dm_plus[i]
            dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / tr_period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.where(tr_sum != 0, (dm_plus_sum / tr_sum) * 100, 0)
    di_minus = np.where(tr_sum != 0, (dm_minus_sum / tr_sum) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    
    adx = np.zeros_like(dx)
    if len(dx) >= tr_period:
        adx[tr_period-1] = np.nanmean(dx[tr_period:2*tr_period]) if 2*tr_period <= len(dx) else np.nan
        for i in range(tr_period, len(dx)):
            adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h RSI for mean reversion signals
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])  # align with index 0
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder smoothing for RSI
    rsi_period = 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    if len(gain) >= rsi_period:
        avg_gain[rsi_period-1] = np.nanmean(gain[1:rsi_period])  # skip index 0 NaN
        avg_loss[rsi_period-1] = np.nanmean(loss[1:rsi_period])
        
        for i in range(rsi_period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50, ADX, and RSI periods
    start_idx = max(50, 14*2)  # EMA50 + ADX/RSI smoothing
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(rsi[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        adx_val = adx_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        rsi_val = rsi[i]
        close_val = close[i]
        
        if position == 0:
            # Regime-based entry
            if adx_val > 25:  # Trending regime
                # Long: price above EMA50
                if close_val > ema_50:
                    signals[i] = 0.25
                    position = 1
                # Short: price below EMA50
                elif close_val < ema_50:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif adx_val < 20:  # Ranging regime
                # Long: RSI oversold
                if rsi_val < 30:
                    signals[i] = 0.25
                    position = 1
                # Short: RSI overbought
                elif rsi_val > 70:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # Transition regime (20 <= ADX <= 25) - no trades
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions
            if adx_val > 25:  # In trend mode - exit on trend reversal
                if close_val < ema_50:
                    signals[i] = 0.0
                    position = 0
            elif adx_val < 20:  # In range mode - exit on RSI normalization
                if rsi_val > 50:  # RSI back to neutral
                    signals[i] = 0.0
                    position = 0
            else:  # Transition regime - exit on either
                if close_val < ema_50 or rsi_val > 50:
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions
            if adx_val > 25:  # In trend mode - exit on trend reversal
                if close_val > ema_50:
                    signals[i] = 0.0
                    position = 0
            elif adx_val < 20:  # In range mode - exit on RSI normalization
                if rsi_val < 50:  # RSI back to neutral
                    signals[i] = 0.0
                    position = 0
            else:  # Transition regime - exit on either
                if close_val > ema_50 or rsi_val < 50:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "6h_RSI_Trend_MeanReversion_Combo_v1"
timeframe = "6h"
leverage = 1.0