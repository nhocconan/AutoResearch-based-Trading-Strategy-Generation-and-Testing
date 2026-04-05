#!/usr/bin/env python3
"""
Experiment #8622: 12h Camarilla Pivot + Volume Spike + Choppiness Regime Filter.
Hypothesis: On 12h timeframe, price reacts to institutional pivot levels (resistance/support)
with volume confirmation. Choppiness filter avoids trades in sideways markets. 
Designed to work in both bull (breakouts) and bear (mean reversion at pivots) regimes.
Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8622_12h_camarilla_pivot_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day's OHLC for Camarilla
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
CHOPPINESS_PERIOD = 14
CHOPPINESS_THRESHOLD = 61.8  # Above = choppy (range), below = trending
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index: higher = more choppy/ranging"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop.fillna(50).values  # neutral when undefined

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (inner strong support/resistance)
    camarilla_h4 = close_1d + (range_1d * 1.1 / 2)  # Resistance 4
    camarilla_l4 = close_1d - (range_1d * 1.1 / 2)  # Support 4
    camarilla_h3 = close_1d + (range_1d * 1.1 / 4)  # Resistance 3
    camarilla_l3 = close_1d - (range_1d * 1.1 / 4)  # Support 3
    
    # Align to 12h timeframe (shifted by 1 day for lookback)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate LTF indicators (12h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Choppiness index for regime filter
    chop = calculate_choppiness(high, low, close, CHOPPINESS_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK, VOLUME_MA_PERIOD, ATR_PERIOD, CHOPPINESS_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(pivot_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Regime filter: avoid choppy markets
        not_choppy = chop[i] <= CHOPPINESS_THRESHOLD  # trending when below threshold
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Price action at Camarilla levels
        near_h4 = abs(close[i] - camarilla_h4_aligned[i]) < (0.1 * atr[i])  # near resistance
        near_l4 = abs(close[i] - camarilla_l4_aligned[i]) < (0.1 * atr[i])  # near support
        near_h3 = abs(close[i] - camarilla_h3_aligned[i]) < (0.1 * atr[i])  # near resistance
        near_l3 = abs(close[i] - camarilla_l3_aligned[i]) < (0.1 * atr[i])  # near support
        
        # Entry logic: 
        # - Near resistance in trending market = short (expect rejection)
        # - Near support in trending market = long (expect bounce)
        # - Breakthrough with volume = follow momentum
        
        if position == 0:
            # Mean reversion at strong levels (H4/L4) - only in non-choppy markets
            if not_choppy and volume_confirmed:
                if near_h4:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                elif near_l4:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Breakout at medium levels (H3/L3) with volume
            elif volume_confirmed:
                if near_h3 and close[i] > camarilla_h3_aligned[i]:
                    signals[i] = SIGNAL_SIZE  # break upward
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                elif near_l3 and close[i] < camarilla_l3_aligned[i]:
                    signals[i] = -SIGNAL_SIZE  # break downward
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals