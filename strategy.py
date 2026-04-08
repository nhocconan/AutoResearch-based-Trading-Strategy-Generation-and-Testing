#!/usr/bin/env python3
# 6h_market_regime_adaptive_v1
# Hypothesis: 6h strategy that adapts to market regime (trending vs ranging) using 1w ADX and 1d ATR ratio.
# In trending regimes (ADX > 25), uses 6h Donchian breakout for trend following.
# In ranging regimes (ADX <= 25), uses 6d RSI mean reversion at Bollinger Bands.
# Volume confirmation filters both regimes to avoid false signals.
# Designed to work in bull/bear markets by capturing trends and fading extremes in ranges.
# Target: 15-30 trades/year with position size 0.25 to minimize fee drag.

name = "6h_market_regime_adaptive_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY ADX FOR REGIME DETECTION ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = wilders_smoothing(tr_1w, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = np.where(atr_1w != 0, dm_plus_smooth / atr_1w * 100, 0)
    di_minus = np.where(atr_1w != 0, dm_minus_smooth / atr_1w * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1w = wilders_smoothing(dx, 14)
    
    # Align weekly ADX to 6h
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # === DAILY ATR RATIO FOR VOLATILITY REGIME (SECONDARY CONFIRMATION) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and ATR (14)
    tr1_d = high_1d - low_1d
    tr2_d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_d[0] = 0
    tr2_d[0] = 0
    tr3_d[0] = 0
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr_14_d = wilders_smoothing(tr_d, 14)
    
    # ATR (50) for ratio
    atr_50_d = wilders_smoothing(tr_d, 50)
    
    # ATR ratio: short-term / long-term volatility
    atr_ratio = np.where(atr_50_d != 0, atr_14_d / atr_50_d, 1.0)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # === 6H INDICATORS ===
    # Donchian Channel (20) for breakouts
    donch_len = 20
    donch_high = np.full_like(close, np.nan)
    donch_low = np.full_like(close, np.nan)
    for i in range(donch_len-1, len(close)):
        donch_high[i] = np.max(high[i-donch_len+1:i+1])
        donch_low[i] = np.min(low[i-donch_len+1:i+1])
    # Forward fill initial values
    for i in range(1, len(close)):
        if np.isnan(donch_high[i]):
            donch_high[i] = donch_high[i-1]
        if np.isnan(donch_low[i]):
            donch_low[i] = donch_low[i-1]
    
    # RSI (6) for mean reversion in ranging markets
    rsi_len = 6
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    def rsi_wilder(gain, loss, period):
        rs = np.full_like(close, np.nan)
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        # First average
        if len(gain) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
        
        # Wilder smoothing
        for i in range(period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_6 = rsi_wilder(gain, loss, rsi_len)
    
    # Bollinger Bands (20, 2.0) for mean reversion
    bb_len = 20
    bb_mult = 2.0
    bb_basis = np.zeros_like(close)
    bb_basis[bb_len-1:] = pd.Series(close).rolling(window=bb_len, min_periods=bb_len).mean()[bb_len-1:].values
    bb_basis[:bb_len-1] = bb_basis[bb_len-1]
    
    bb_dev = bb_mult * pd.Series(close).rolling(window=bb_len, min_periods=bb_len).std()[bb_len-1:].values
    bb_dev[:bb_len-1] = bb_dev[bb_len-1]
    
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = np.zeros_like(volume)
    vol_ma[19:] = pd.Series(volume).rolling(window=20, min_periods=20).mean()[19:].values
    vol_ma[:19] = vol_ma[19]
    
    # Start from sufficient lookback
    start_idx = max(donch_len, rsi_len, bb_len, 20) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(rsi_6[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.3 * vol_ma[i]
        
        # Regime determination: Trending if weekly ADX > 25 AND ATR ratio > 0.8 (not low volatility)
        is_trending = adx_1w_aligned[i] > 25 and atr_ratio_aligned[i] > 0.8
        
        if position == 1:  # Long position
            if is_trending:
                # Exit trend following: price re-enters Donchian channel or ADX weakens
                if close[i] <= donch_high[i] or adx_1w_aligned[i] < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                # Exit mean reversion: RSI returns to neutral or hits opposite band
                if rsi_6[i] >= 50 or close[i] >= bb_lower[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            if is_trending:
                # Exit trend following: price re-enters Donchian channel or ADX weakens
                if close[i] >= donch_low[i] or adx_1w_aligned[i] < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                # Exit mean reversion: RSI returns to neutral or hits opposite band
                if rsi_6[i] <= 50 or close[i] <= bb_upper[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat, look for entry
            if is_trending:
                # Trending regime: Donchian breakout with volume
                # Long: break above upper band
                if close[i] > donch_high[i] and volume_filter:
                    position = 1
                    signals[i] = 0.25
                # Short: break below lower band
                elif close[i] < donch_low[i] and volume_filter:
                    position = -1
                    signals[i] = -0.25
            else:
                # Ranging regime: RSI mean reversion at Bollinger Bands
                # Long: RSI oversold (<30) and price at/below lower BB
                if rsi_6[i] < 30 and close[i] <= bb_lower[i] and volume_filter:
                    position = 1
                    signals[i] = 0.25
                # Short: RSI overbought (>70) and price at/above upper BB
                elif rsi_6[i] > 70 and close[i] >= bb_upper[i] and volume_filter:
                    position = -1
                    signals[i] = -0.25
    
    return signals