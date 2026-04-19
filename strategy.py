#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d regime filter
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - 1d ADX(14) defines regime: ADX > 25 = trending, ADX < 20 = ranging
# - In trending regime: Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling
# - In ranging regime: Mean reversion at Bollinger Bands (20,2) from 6h
# - Volume confirmation: 6h volume > 1.5x 20-period average
# - Position size: 0.25 to manage drawdown in volatile markets
# - Designed to work in both bull and bear markets by adapting to regime
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "6h_ElderRay_1dADXRegime_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX(14) for regime detection
    # Calculate +DM, -DM, TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            result[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period = 14
    if len(tr) >= period:
        atr_1d = wilders_smoothing(tr, period)
        plus_di_1d = 100 * wilders_smoothing(plus_dm, period) / atr_1d
        minus_di_1d = 100 * wilders_smoothing(minus_dm, period) / atr_1d
        dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
        adx_1d = wilders_smoothing(dx_1d, period)
    else:
        adx_1d = np.full_like(tr, np.nan)
    
    # Prepend NaN to match original length
    adx_1d_full = np.full(len(close_1d), np.nan)
    if len(adx_1d) > 0:
        adx_1d_full[13:] = adx_1d
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_full)
    
    # 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # 6h Bollinger Bands (20,2) for ranging regime
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # 6h volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute signals array
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime: ADX > 25 = trending, ADX < 20 = ranging
        if adx_1d_aligned[i] > 25:
            # Trending regime
            if position == 0:
                # Long when Bull Power > 0 and rising (current > previous)
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short when Bear Power < 0 and falling (current < previous)
                elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long when Bull Power becomes negative
                if bull_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short when Bear Power becomes positive
                if bear_power[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        elif adx_1d_aligned[i] < 20:
            # Ranging regime - mean reversion at Bollinger Bands
            if position == 0:
                # Long at lower band
                if close[i] <= bb_lower[i]:
                    signals[i] = 0.25
                    position = 1
                # Short at upper band
                elif close[i] >= bb_upper[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long at middle or upper band
                if close[i] >= sma_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short at middle or lower band
                if close[i] <= sma_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Transition zone (ADX between 20-25) - hold or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals