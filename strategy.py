#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) mean reversion with 12h ADX (14) regime filter and volume confirmation
# Long when Williams %R < -80 (oversold) + 12h ADX < 25 (range/weak trend) + volume > 1.5x 20-period avg
# Short when Williams %R > -20 (overbought) + 12h ADX < 25 + volume > 1.5x 20-period avg
# Williams %R identifies extreme reversals; low ADX filters out strong trends where mean reversion fails
# Volume confirmation avoids low-liquidity false signals. Discrete sizing (0.25) controls drawdown.
# Designed for ranging markets (2025-2026 bear/range) but works in bull via mean reversion in pullbacks.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: ADX(14) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilde_rma(data, period):
        """Wilder's RMA (same as EMA with alpha=1/period)"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # first value is simple average
        result[period-1] = np.nanmean(data[:period])
        # subsequent values: Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_12h = wilde_rma(tr, 14)
    plus_di_12h = 100 * wilde_rma(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilde_rma(minus_dm, 14) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilde_rma(dx_12h, 14)
    
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Williams %R (14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_6h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r_6h = -100 * (highest_high_6h - close) / (highest_high_6h - lowest_low_6h)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(14, 20) + 10  # Williams %R(14) + volume(20) + ADX buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r_6h[i]) or np.isnan(adx_12h_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Williams %R < -80 (oversold)
        # 2. 12h ADX < 25 (range/weak trend - favors mean reversion)
        # 3. Volume confirmation
        if (williams_r_6h[i] < -80) and \
           (adx_12h_aligned[i] < 25) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R > -20 (overbought)
        # 2. 12h ADX < 25 (range/weak trend - favors mean reversion)
        # 3. Volume confirmation
        elif (williams_r_6h[i] > -20) and \
             (adx_12h_aligned[i] < 25) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR14_12hADX25_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0