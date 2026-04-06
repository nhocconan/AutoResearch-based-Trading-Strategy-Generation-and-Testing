#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot + volume confirmation + 1d trend filter.
# Camarilla pivot levels (R3/S3, R4/S4) provide high-probability reversal/breakout zones.
# In ranging markets: fade at R3/S3 (mean reversion). In trending markets: breakout continuation at R4/S4.
# Use 1d EMA(50) for trend filter: only take R3/S3 fades in ranging (ADX<25) and R4/S4 breakouts in trending (ADX>=25).
# Volume confirmation filters low-quality signals. Target: 15-35 trades/year by requiring confluence.
# Works in bull/bear via regime adaptation: mean revert in range, breakout in trend.

name = "exp_13647_6h_camarilla_pivot_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_LOOKBACK = 20  # Use previous 20 periods for pivot calculation
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    # Calculate Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
    di_minus = 100 * dm_minus_smooth / (tr_smooth + 1e-10)
    
    # Calculate DX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    
    # Calculate ADX
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for given high, low, close
    Returns: (R4, R3, R2, R1, PP, S1, S2, S3, S4)
    """
    typical = (high + low + close) / 3
    range_val = high - low
    
    # Camarilla formulas
    R4 = close + range_val * 1.1 / 2
    R3 = close + range_val * 1.1 / 4
    R2 = close + range_val * 1.1 / 6
    R1 = close + range_val * 1.1 / 12
    PP = typical
    S1 = close - range_val * 1.1 / 12
    S2 = close - range_val * 1.1 / 6
    S3 = close - range_val * 1.1 / 4
    S4 = close - range_val * 1.1 / 2
    
    return R4, R3, R2, R1, PP, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot and trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX for trend filter
    adx = calculate_adx(high_1d, low_1d, close_1d, ADX_PERIOD)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_levels = []
    for i in range(len(close_1d)):
        R4, R3, R2, R1, PP, S1, S2, S3, S4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        camarilla_levels.append((R4, R3, R2, R1, PP, S1, S2, S3, S4))
    
    # Unpack levels
    R4_1d = np.array([x[0] for x in camarilla_levels])
    R3_1d = np.array([x[1] for x in camarilla_levels])
    R2_1d = np.array([x[2] for x in camarilla_levels])
    R1_1d = np.array([x[3] for x in camarilla_levels])
    PP_1d = np.array([x[4] for x in camarilla_levels])
    S1_1d = np.array([x[5] for x in camarilla_levels])
    S2_1d = np.array([x[6] for x in camarilla_levels])
    S3_1d = np.array([x[7] for x in camarilla_levels])
    S4_1d = np.array([x[8] for x in camarilla_levels])
    
    # Align Camarilla levels to 6h timeframe
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    R2_1d_aligned = align_htf_to_ltf(prices, df_1d, R2_1d)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    PP_1d_aligned = align_htf_to_ltf(prices, df_1d, PP_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(CAMARILLA_LOOKBACK, ADX_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(R4_1d_aligned[i]) or np.isnan(S4_1d_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend determination from ADX
        trending = adx_aligned[i] >= ADX_TREND_THRESHOLD
        ranging = adx_aligned[i] < ADX_TREND_THRESHOLD
        
        # Camarilla-based signals
        # In ranging market: fade at R3/S3 (mean reversion)
        # In trending market: breakout continuation at R4/S4
        
        # Long signals
        long_signal = False
        if ranging and volume_ok:
            # Fade at S3: price rejects S3 and moves back up
            if i > 0 and low[i] <= S3_1d_aligned[i] and close[i] > S3_1d_aligned[i]:
                long_signal = True
        elif trending and volume_ok:
            # Breakout above R4: continuation
            if i > 0 and close[i-1] <= R4_1d_aligned[i] and close[i] > R4_1d_aligned[i]:
                long_signal = True
        
        # Short signals
        short_signal = False
        if ranging and volume_ok:
            # Fade at R3: price rejects R3 and moves back down
            if i > 0 and high[i] >= R3_1d_aligned[i] and close[i] < R3_1d_aligned[i]:
                short_signal = True
        elif trending and volume_ok:
            # Breakdown below S4: continuation
            if i > 0 and close[i-1] >= S4_1d_aligned[i] and close[i] < S4_1d_aligned[i]:
                short_signal = True
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on opposite signal or stop loss
            # Exit conditions: reverse of entry
            exit_signal = False
            if ranging:
                # Exit long if price reaches R3 (profit target) or rejects S3 again
                if i > 0 and high[i] >= R3_1d_aligned[i]:
                    exit_signal = True
            else:  # trending
                # Exit long if price breaks below R4 (failed breakout) or reaches S4 (extended target)
                if i > 0 and (close[i] < R4_1d_aligned[i] or low[i] <= S4_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite signal or stop loss
            # Exit conditions: reverse of entry
            exit_signal = False
            if ranging:
                # Exit short if price reaches S3 (profit target) or rejects R3 again
                if i > 0 and low[i] <= S3_1d_aligned[i]:
                    exit_signal = True
            else:  # trending
                # Exit short if price breaks above S4 (failed breakdown) or reaches R4 (extended target)
                if i > 0 and (close[i] > S4_1d_aligned[i] or high[i] >= R4_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals