#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour timeframe using weekly pivot points for trend direction and daily ATR for volatility filtering.
# Weekly pivot levels (PP, R1, S1, R2, S2) determine bias: price above PP = long bias, below PP = short bias.
# Entries occur on pullbacks to daily VWAP with volume confirmation (>1.5x 20-period average).
# Weekly timeframe provides strong trend filter reducing whipsaws, daily VWAP provides precise entry timing.
# Designed for 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

name = "exp_13715_6h_weekly_pivot_daily_vwap_vol"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 5  # bars to confirm pivot level respect
VWAP_PERIOD = 20    # for VWAP calculation
VOLUME_MA_PERIOD = 20  # for volume filter
VOLUME_THRESHOLD = 1.5  # volume must be 1.5x average
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_pivots(high, low, close):
    """Calculate weekly pivot points: PP, R1, S1, R2, S2"""
    pp = (high + low + close) / 3.0
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    return pp, r1, s1, r2, s2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_vwap(high, low, close, volume, period):
    """Calculate VWAP over specified period"""
    typical_price = (high + low + close) / 3.0
    vwap = pd.Series(typical_price * volume).rolling(window=period, min_periods=period).sum() / \
           pd.Series(volume).rolling(window=period, min_periods=period).sum()
    return vwap.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot points ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points
    weekly_pp, weekly_r1, weekly_s1, weekly_r2, weekly_s2 = calculate_pivots(
        weekly_high, weekly_low, weekly_close
    )
    
    # Align weekly pivot data to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    
    # Load daily data for VWAP ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    daily_volume = df_daily['volume'].values
    
    # Calculate daily VWAP
    daily_vwap = calculate_vwap(daily_high, daily_low, daily_close, daily_volume, VWAP_PERIOD)
    vwap_aligned = align_htf_to_ltf(prices, df_daily, daily_vwap)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK, VWAP_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(vwap_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Price relative to weekly pivot and VWAP
        above_pp = close[i] > pp_aligned[i]
        below_pp = close[i] < pp_aligned[i]
        near_vwap = np.abs(close[i] - vwap_aligned[i]) < (0.5 * atr[i])  # within 0.5 ATR of VWAP
        
        # Pullback definition: price near VWAP in direction of weekly trend
        long_setup = above_pp and near_vwap and volume_ok
        short_setup = below_pp and near_vwap and volume_ok
        
        # Generate signals
        if position == 0:
            if long_setup:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_setup:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long if price breaks below weekly S1 or stops hit
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short if price breaks above weekly R1 or stops hit
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals