#!/usr/bin/env python3
"""
Experiment #8631: 6h Camarilla Pivot + Volume Spike + Regime Filter
Hypothesis: Camarilla pivot levels from daily timeframe provide institutional support/resistance.
In ranging markets (Choppiness > 61.8), fade at R3/S3 levels with volume confirmation.
In trending markets (Choppiness < 38.2), breakout continuation at R4/S4 levels.
Volume spike (>2x 20-period mean) confirms institutional participation.
Target: 75-150 trades over 4 years (19-38/year) to balance frequency and edge.
Works in bull/bear via regime adaptation.
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8631_6h_camarilla_pivot_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC
CHOPPINESS_PERIOD = 14
CHOPPINESS_THRESHOLD_TREND = 38.2
CHOPPINESS_THRESHOLD_RANGE = 61.8
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_choppiness(high, low, close, period):
    """Choppiness Index: higher = ranging, lower = trending"""
    atr = []
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), 
                    np.abs(low - np.roll(close, 1)))
    for i in range(len(close)):
        if i < period:
            atr.append(np.nan)
        else:
            sum_tr = np.nansum(tr[i-period+1:i+1])
            highest = np.nanmax(high[i-period+1:i+1])
            lowest = np.nanmin(low[i-period+1:i+1])
            if highest == lowest:
                chop = 50
            else:
                chop = 100 * np.log10(sum_tr / (highest - lowest)) / np.log10(period)
            atr.append(chop)
    return np.array(atr)

def calculate_atr(high, low, close, period):
    """ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # S4 = C - ((H-L) * 1.1/2), S3 = C - ((H-L) * 1.1/4), etc.
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels (using previous day's values)
    camarilla_r4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align to 6s timeframe (shifted by 1 day for lookback)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Choppiness index (regime filter)
    chop = calculate_choppiness(high, low, close, CHOPPINESS_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(CHOPPINESS_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or \
           np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]):
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
        
        # Determine market regime
        is_ranging = chop[i] > CHOPPINESS_THRESHOLD_RANGE
        is_trending = chop[i] < CHOPPINESS_THRESHOLD_TREND
        
        # Volume confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Initialize signal
        signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
        
        # Entry logic based on regime
        if position == 0:  # Only enter new positions when flat
            if is_ranging and volume_spike:
                # Ranging market: fade at R3/S3
                if close[i] <= r3_aligned[i]:  # Sell at R3 resistance
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                elif close[i] >= s3_aligned[i]:  # Buy at S3 support
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif is_trending and volume_spike:
                # Trending market: breakout continuation at R4/S4
                if close[i] >= r4_aligned[i]:  # Buy breakout above R4
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                elif close[i] <= s4_aligned[i]:  # Sell breakdown below S4
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
    
    return signals