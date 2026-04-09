#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band breakout with 1d volume confirmation and ADX trend filter
# - Uses 12h Bollinger Bands (20, 2.0) for breakout entries
# - Volume confirmation: 12h volume > 1.5x 20-period average to ensure breakout strength
# - Trend filter: 1d ADX > 25 to ensure we only trade in trending markets (avoid whipsaws in range)
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in both bull/bear: Bollinger Bands adapt to volatility, ADX filter ensures we only trade when trend is strong

name = "12h_1d_bb_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) for trend filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = WilderSmooth(tr_1d, 14)
    plus_dm_14 = WilderSmooth(plus_dm, 14)
    minus_dm_14 = WilderSmooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di_14 = np.where(atr_1d > 0, 100 * plus_dm_14 / atr_1d, 0)
    minus_di_14 = np.where(atr_1d > 0, 100 * minus_dm_14 / atr_1d, 0)
    dx_14 = np.where((plus_di_14 + minus_di_14) > 0, 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14), 0)
    adx_1d = WilderSmooth(dx_14, 14)
    
    # Align ADX to 12h timeframe (completed 1d bar only)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Bollinger Bands (20, 2.0)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (2.0 * bb_std)
    bb_lower = bb_middle - (2.0 * bb_std)
    
    # 12h volume > 1.5x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    # 12h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(atr[i]) or
            atr[i] <= 0 or
            adx_1d_aligned[i] < 25):  # ADX trend filter - only trade when trending
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high OR touches BB middle (mean reversion)
            if low[i] <= highest_since_entry - (2.0 * atr[i]) or \
               low[i] <= bb_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low OR touches BB middle (mean reversion)
            if high[i] >= lowest_since_entry + (2.0 * atr[i]) or \
               high[i] >= bb_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Bollinger Band breakout with volume confirmation
            # Long: price breaks above BB upper AND volume spike
            if high[i] >= bb_upper[i] and volume_spike[i]:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: price breaks below BB lower AND volume spike
            elif low[i] <= bb_lower[i] and volume_spike[i]:
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals