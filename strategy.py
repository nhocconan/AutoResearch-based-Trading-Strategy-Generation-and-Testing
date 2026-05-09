#!/usr/bin/env python3
"""
6h_MarketFacets_Momentum_Regime
Hypothesis: Combines multi-faceted momentum (price acceleration, volume surge, and volatility breakout) with a regime filter (trending vs ranging) to capture high-probability moves in both bull and bear markets. Uses 1w trend for primary direction, 1d for entry signals, and 6s for execution. Designed for low trade frequency (15-35/year) with high win rate by requiring confluence of multiple uncorrelated signals.
"""
name = "6h_MarketFacets_Momentum_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for primary trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA(50) for trend
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 48) / 50
    
    # Align 1w EMA to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ROC(10) for momentum
    roc_10_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 10:
        roc_10_1d[9:] = (close_1d[9:] - close_1d[:-9]) / close_1d[:-9] * 100
    
    # Calculate 1d volume spike (current vs 20-period average)
    vol_ma_20_1d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        vol_ma_20_1d[19] = np.mean(volume_1d[0:20])
        for i in range(20, len(volume_1d)):
            vol_ma_20_1d[i] = (vol_ma_20_1d[i-1] * 19 + volume_1d[i]) / 20
    
    volume_spike_1d = np.full_like(volume_1d, np.nan)
    valid_vol = (~np.isnan(vol_ma_20_1d)) & (vol_ma_20_1d != 0)
    volume_spike_1d[valid_vol] = volume_1d[valid_vol] / vol_ma_20_1d[valid_vol]
    
    # Calculate 1d ATR(14) for volatility breakout
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with index
    atr_14_1d = np.full_like(tr_1d, np.nan)
    if len(tr_1d) >= 14:
        atr_14_1d[13] = np.nanmean(tr_1d[1:15])  # skip first NaN
        for i in range(14, len(tr_1d)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Volatility breakout: current range > 1.5 * ATR
    daily_range = high_1d - low_1d
    vol_breakout = np.full_like(daily_range, np.nan)
    valid_atr = ~np.isnan(atr_14_1d)
    vol_breakout[valid_atr] = daily_range[valid_atr] > (atr_14_1d[valid_atr] * 1.5)
    
    # Align 1d indicators to 6h timeframe
    roc_10_1d_aligned = align_htf_to_ltf(prices, df_1d, roc_10_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    vol_breakout_aligned = align_htf_to_ltf(prices, df_1d, vol_breakout.astype(float))
    
    # Regime filter: 1d ADX(14) to distinguish trending vs ranging
    # Calculate +DM, -DM, TR
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    up_move = np.concatenate([[np.nan], up_move])
    down_move = np.concatenate([[np.nan], down_move])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[1:period+1])  # skip first NaN
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_1d_smooth = wilder_smooth(tr_1d, 14)
    plus_di_14 = np.full_like(tr_1d, np.nan)
    minus_di_14 = np.full_like(tr_1d, np.nan)
    dx_14 = np.full_like(tr_1d, np.nan)
    
    if len(tr_1d_smooth) >= 14:
        valid_tr = ~np.isnan(tr_1d_smooth) & (tr_1d_smooth != 0)
        plus_di_14[valid_tr] = (wilder_smooth(plus_dm, 14)[valid_tr] / tr_1d_smooth[valid_tr]) * 100
        minus_di_14[valid_tr] = (wilder_smooth(minus_dm, 14)[valid_tr] / tr_1d_smooth[valid_tr]) * 100
        dx_14 = np.where((plus_di_14 + minus_di_14) != 0,
                         np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100,
                         0)
    
    adx_14 = wilder_smooth(dx_14, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(roc_10_1d_aligned[i]) or \
           np.isnan(volume_spike_1d_aligned[i]) or np.isnan(vol_breakout_aligned[i]) or \
           np.isnan(adx_14_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime: ADX > 25 indicates trending market
        is_trending = adx_14_aligned[i] > 25
        
        if position == 0:
            # Enter long: Price above weekly EMA50 AND bullish momentum AND volume spike AND volatility breakout
            if (close[i] > ema_50_1w_aligned[i] and
                roc_10_1d_aligned[i] > 2.0 and  # positive momentum
                volume_spike_1d_aligned[i] > 2.0 and  # volume surge
                vol_breakout_aligned[i] and  # volatility expansion
                is_trending):
                signals[i] = 0.25
                position = 1
            # Enter short: Price below weekly EMA50 AND bearish momentum AND volume spike AND volatility breakout
            elif (close[i] < ema_50_1w_aligned[i] and
                  roc_10_1d_aligned[i] < -2.0 and  # negative momentum
                  volume_spike_1d_aligned[i] > 2.0 and  # volume surge
                  vol_breakout_aligned[i] and  # volatility expansion
                  is_trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Loss of momentum or trend reversal
            if (roc_10_1d_aligned[i] < 0 or  # momentum faded
                close[i] < ema_50_1w_aligned[i]):  # trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Loss of momentum or trend reversal
            if (roc_10_1d_aligned[i] > 0 or  # momentum faded
                close[i] > ema_50_1w_aligned[i]):  # trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals