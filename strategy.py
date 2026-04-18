#!/usr/bin/env python3
"""
4h_Range_MeanReversion_ATR_Pivot
Hypothesis: In range-bound markets, price tends to revert to the mean near daily pivots.
Use 4h timeframe with daily ATR-scaled pivot levels (R1/S1) for mean reversion entries.
Long when price touches or slightly breaks below S1 with bullish reversal candle.
Short when price touches or slightly breaks above R1 with bearish reversal candle.
Exit when price crosses the daily pivot (mean) or ATR-based stop is hit.
Designed for 4h to achieve ~20-40 trades/year with low turnover and high win rate in ranging markets.
Works in both bull and bear by avoiding strong trends via ADX filter on higher timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for dynamic pivot width
    def calculate_atr(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period] = np.nanmean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Calculate daily pivot points (standard)
    pivot = np.full_like(high_1d, np.nan)
    R1 = np.full_like(high_1d, np.nan)
    S1 = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        pivot[i] = (prev_high + prev_low + prev_close) / 3.0
        range_ = prev_high - prev_low
        R1[i] = pivot[i] + range_
        S1[i] = pivot[i] - range_
    
    # Get 12h data for trend filter (ADX to avoid strong trends)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Align all data to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    atr_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    adx_4h = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 4h ATR for entry/exit triggers
    atr_4h_raw = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14, 14) + 5  # Buffer for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_4h[i]) or np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(adx_4h[i]) or np.isnan(atr_4h_raw[i])):
            signals[i] = 0.0
            continue
        
        # Range regime filter: avoid strong trends
        range_regime = adx_4h[i] < 25
        
        if position == 0:
            # Long: price near S1 with bullish reversal
            near_s1 = low[i] <= S1_4h[i] + 0.1 * atr_4h_raw[i]  # Allow small penetration
            bullish_reversal = close[i] > open_prices[i] and close[i] > low[i]  # Bullish candle
            
            if near_s1 and bullish_reversal and range_regime:
                signals[i] = 0.25
                position = 1
            
            # Short: price near R1 with bearish reversal
            near_r1 = high[i] >= R1_4h[i] - 0.1 * atr_4h_raw[i]
            bearish_reversal = close[i] < open_prices[i] and close[i] < high[i]
            
            if near_r1 and bearish_reversal and range_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses pivot (mean) or ATR stop
            exit_signal = close[i] > pivot_4h[i]  # Reached mean
            stop_loss = close[i] < entry_price - 1.5 * atr_4h_raw[i] if 'entry_price' in locals() else False
            
            if exit_signal or stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses pivot (mean) or ATR stop
            exit_signal = close[i] < pivot_4h[i]  # Reached mean
            stop_loss = close[i] > entry_price + 1.5 * atr_4h_raw[i] if 'entry_price' in locals() else False
            
            if exit_signal or stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Helper function for ADX calculation (defined inside to avoid redefinition issues)
def calculate_adx(high, low, close, period=14):
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    if len(tr) >= period:
        atr[period] = np.nanmean(tr[1:period+1])
        dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
        
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    di_plus = np.full_like(dm_plus_smooth, np.nan)
    di_minus = np.full_like(dm_minus_smooth, np.nan)
    valid = ~np.isnan(atr) & (atr != 0)
    di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
    di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
    
    dx = np.full_like(di_plus, np.nan)
    dx_valid = ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
    dx[dx_valid] = 100 * np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])
    
    adx = np.full_like(dx, np.nan)
    if len(dx) >= 2*period:
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

name = "4h_Range_MeanReversion_ATR_Pivot"
timeframe = "4h"
leverage = 1.0