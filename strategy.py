#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI(14) mean reversion with 4-hour ADX(14) trend filter and 1-day volume filter.
# Uses 4-hour ADX to identify trending vs ranging markets: ADX < 25 = range (mean revert), ADX > 25 = trend (avoid).
# 1-day volume filter ensures trades occur during high liquidity periods.
# RSI extremes (<30 for long, >70 for short) provide mean reversion signals in ranging markets.
# Designed for low trade frequency (target 60-150 over 4 years) to minimize fee drag.
# Works in bull/bear by avoiding counter-trend trades in strong trends.

name = "1h_rsi14_4h_adx14_1d_vol_meanrev_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period RSI
    rsi = np.full(n, np.nan)
    if n >= 14:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        if n > 14:
            avg_gain[14] = np.mean(gain[1:15])
            avg_loss[14] = np.mean(loss[1:15])
            for i in range(15, n):
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:14] = np.nan
    
    # 14-period ADX on 4-hour timeframe
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range and Directional Movement
    tr_4h = np.maximum(
        high_4h[1:] - low_4h[1:],
        np.abs(high_4h[1:] - close_4h[:-1]),
        np.abs(low_4h[1:] - close_4h[:-1])
    )
    dm_plus_4h = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                          np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus_4h = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                           np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (alpha = 1/14)
    tr_14_4h = np.full(len(tr_4h), np.nan)
    dm_plus_14_4h = np.full(len(tr_4h), np.nan)
    dm_minus_14_4h = np.full(len(tr_4h), np.nan)
    if len(tr_4h) >= 14:
        tr_14_4h[13] = np.sum(tr_4h[:14])
        dm_plus_14_4h[13] = np.sum(dm_plus_4h[:14])
        dm_minus_14_4h[13] = np.sum(dm_minus_4h[:14])
        for i in range(14, len(tr_4h)):
            tr_14_4h[i] = tr_14_4h[i-1] - (tr_14_4h[i-1] / 14) + tr_4h[i]
            dm_plus_14_4h[i] = dm_plus_14_4h[i-1] - (dm_plus_14_4h[i-1] / 14) + dm_plus_4h[i]
            dm_minus_14_4h[i] = dm_minus_14_4h[i-1] - (dm_minus_14_4h[i-1] / 14) + dm_minus_4h[i]
    
    # Directional Indicators
    di_plus_4h = np.full(len(tr_4h), np.nan)
    di_minus_4h = np.full(len(tr_4h), np.nan)
    dx_4h = np.full(len(tr_4h), np.nan)
    mask = tr_14_4h != 0
    di_plus_4h[mask] = 100 * dm_plus_14_4h[mask] / tr_14_4h[mask]
    di_minus_4h[mask] = 100 * dm_minus_14_4h[mask] / tr_14_4h[mask]
    dx_4h[mask] = 100 * np.abs(di_plus_4h[mask] - di_minus_4h[mask]) / (di_plus_4h[mask] + di_minus_4h[mask])
    
    # ADX: smoothed DX
    adx_4h = np.full(len(dx_4h), np.nan)
    if len(dx_4h) >= 14:
        valid_dx = dx_4h[~np.isnan(dx_4h)]
        if len(valid_dx) >= 14:
            start_idx = len(dx_4h) - len(valid_dx)
            adx_4h[start_idx + 13] = np.mean(valid_dx[:14])
            for i in range(start_idx + 14, len(dx_4h)):
                if not np.isnan(dx_4h[i]):
                    adx_4h[i] = (adx_4h[i-1] * 13 + dx_4h[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 20-period volume moving average on 1-day timeframe
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_ma_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_ma_1d[i] = (volume_1d[i] + vol_ma_1d[i-1] * 19) / 20
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session and volume filters
        hour = hours[i]
        session_filter = 8 <= hour <= 20
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # Range condition: ADX < 25 (not trending)
        range_filter = adx_aligned[i] < 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI returns to neutral (50) or stoploss hit
            if (rsi[i] >= 50 or
                close[i] < entry_price - 2.0 * (high[i] - low[i]).mean()):  # simplified volatility
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI returns to neutral (50) or stoploss hit
            if (rsi[i] <= 50 or
                close[i] > entry_price + 2.0 * (high[i] - low[i]).mean()):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries in ranging markets during session
            if session_filter and volume_filter and range_filter:
                # Long: RSI oversold (<30)
                if rsi[i] < 30:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: RSI overbought (>70)
                elif rsi[i] > 70:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals