#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# Long when: Price breaks above 20-period 12h Donchian upper band AND 1d volume > 1.3x 20-period average AND chop > 61.8 (range regime)
# Short when: Price breaks below 20-period 12h Donchian lower band AND 1d volume > 1.3x 20-period average AND chop > 61.8 (range regime)
# Exit when price touches opposite Donchian band or ATR-based stoploss (2.5x ATR)
# Donchian channels provide clear breakout levels in ranging markets
# Volume spike confirms institutional participation
# Chop regime filter (>61.8) ensures we trade in ranging markets where mean reversion works
# Target: 60-120 total trades over 4 years (15-30/year) with discrete sizing 0.25

name = "12h_Donchian20_1dVolumeSpike_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    # Upper band = highest high over last 20 periods
    # Lower band = lowest low over last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR for chop and stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR14) / (n * (max(high) - min(low)))) / log10(n)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    # Avoid division by zero and invalid values
    chop = 100 * (np.log10(sum_atr_14 / (14 * range_14)) / np.log10(14))
    chop = np.where((range_14 == 0) | np.isnan(chop) | (range_14 <= 0), 50, chop)
    chop_regime = chop > 61.8  # Range regime
    
    # Calculate 1d volume spike (current volume > 1.3x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.3 * vol_ma_20)
    
    # Align 1d indicators to 12h timeframe
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    atr_multiplier = 2.5  # ATR stoploss multiplier
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(chop_regime_aligned[i]) or np.isnan(vol_spike_aligned[i]) or
            np.isnan(atr_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get aligned 1d values
        vol_cond = bool(vol_spike_aligned[i])
        chop_cond = bool(chop_regime_aligned[i])
        
        if position == 0:
            # Long: Break above Donchian upper in range regime with volume spike
            if close[i] > donchian_upper[i] and vol_cond and chop_cond:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower in range regime with volume spike
            elif close[i] < donchian_lower[i] and vol_cond and chop_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit conditions for long
            # 1. Price touches Donchian lower band (mean reversion target)
            # 2. ATR-based stoploss: price < highest high since entry - atr_multiplier * current ATR
            exit_long = False
            if close[i] <= donchian_lower[i]:
                exit_long = True
            else:
                # Track highest high since entry for trailing stop logic
                # Simple close-based stop: if price drops below entry point minus ATR buffer
                # We approximate by checking if current price is significantly below recent highs
                if i >= 20:  # Need enough lookback
                    recent_high = np.max(high[max(0, i-20):i+1])
                    if close[i] < recent_high - (atr_multiplier * atr_1d[i]):
                        exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            # 1. Price touches Donchian upper band (mean reversion target)
            # 2. ATR-based stoploss: price > lowest low since entry + atr_multiplier * current ATR
            exit_short = False
            if close[i] >= donchian_upper[i]:
                exit_short = True
            else:
                # Track lowest low since entry for trailing stop logic
                if i >= 20:
                    recent_low = np.min(low[max(0, i-20):i+1])
                    if close[i] > recent_low + (atr_multiplier * atr_1d[i]):
                        exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals