#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume confirmation and choppiness regime.
- Entry: Long when price breaks above R3 AND 1d volume > 1.5x 20-period average AND chop < 61.8 (trending);
         Short when price breaks below S3 AND same conditions.
- Exit: Opposite Camarilla level (R3/S3) break or choppiness regime shifts to chop (>61.8).
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla provides intraday support/resistance, volume confirms conviction, chop filter avoids ranging markets.
- Works in bull markets (buy R3 breakouts) and bear markets (sell S3 breakdowns).
- Estimated trades: ~100 total over 4 years (~25/year) based on breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period."""
    # Camarilla levels based on previous period's high, low, close
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    rng = high - low
    r3 = close + (rng * 1.1 / 4)
    s3 = close - (rng * 1.1 / 4)
    return r3, s3

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    # True Range
    tr1 = pd.Series(high).rolling(window=1).max() - pd.Series(low).rolling(window=1).min()
    tr2 = abs(pd.Series(high).rolling(window=1).max() - pd.Series(close).rolling(window=1).shift(1))
    tr3 = abs(pd.Series(low).rolling(window=1).min() - pd.Series(close).rolling(window=1).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of True Range over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    max_hh = pd.Series(high).rolling(window=period, min_periods=period).max()
    min_ll = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (max_hh - min_ll)) / np.log10(period)
    # Handle division by zero when max_hh == min_ll
    chop = np.where((max_hh - min_ll) == 0, 0, chop)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d HTF indicators for regime and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Camarilla levels (for context, though we use 12h for entries)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    r3_1d, s3_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # 1d volume confirmation: volume > 1.5x 20-period average
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (vol_ma_1d * 1.5)
    
    # 1d choppiness regime: chop < 61.8 = trending (favor breakouts)
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d, 14)
    chop_regime_1d = chop_1d < 61.8  # True when trending
    
    # Align 1d indicators to 12h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d, additional_delay_bars=1)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_1d, additional_delay_bars=1)
    
    # Calculate 12h Camarilla levels for entry signals
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    r3_12h, s3_12h = calculate_camarilla(high_12h, low_12h, close_12h)
    
    # Breakout signals: price closes above R3 or below S3
    breakout_above = close_12h > r3_12h
    breakout_below = close_12h < s3_12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for calculations (max period 20 for volume MA)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(volume_spike_aligned[i]) or np.isnan(chop_regime_aligned[i]) or
            np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or
            np.isnan(volume[i]) or np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite breakout OR chop regime shifts to ranging (>61.8)
        if position != 0:
            # Exit long: price breaks below S3 OR chop becomes ranging
            if position == 1:
                if breakout_below[i] or not chop_regime_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above R3 OR chop becomes ranging
            elif position == -1:
                if breakout_above[i] or not chop_regime_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: breakout in direction of volume spike AND trending regime
        if position == 0:
            # Long: breakout above R3 AND volume spike AND trending chop
            if breakout_above[i] and volume_spike_aligned[i] and chop_regime_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 AND volume spike AND trending chop
            elif breakout_below[i] and volume_spike_aligned[i] and chop_regime_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dVolChop_Regime_v1"
timeframe = "12h"
leverage = 1.0