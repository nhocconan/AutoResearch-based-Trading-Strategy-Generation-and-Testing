#!/usr/bin/env python3
"""
Experiment #9597: 4h Donchian(20) Breakout + Volume Confirmation + Regime Filter.
Hypothesis: Donchian channel breakouts capture momentum in trending markets, while
volume confirmation filters false breakouts and regime filter (Choppiness) adapts
to market conditions. Works in bull (breakouts above upper band) and bear 
(breakdowns below lower band) with reduced frequency in ranging markets.
Targets 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9597_4h_donchian20_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
CHOPPINESS_PERIOD = 14
CHOPPINESS_THRESHOLD = 38.2  # Below = trending, Above = ranging
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
    """Calculate Choppiness Index"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    chop = np.where(
        (highest_high - lowest_low) != 0,
        100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period),
        50
    )
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for regime filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Choppiness for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d, CHOPPINESS_PERIOD)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate LTF indicators (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(chop_1d_aligned[i]):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Regime filter: Chop < 38.2 = trending (favor breakouts), Chop > 38.2 = ranging (favor mean reversion)
        trending_market = chop_1d_aligned[i] < CHOPPINESS_THRESHOLD
        ranging_market = chop_1d_aligned[i] >= CHOPPINESS_THRESHOLD
        
        # Breakout signals (favor in trending markets)
        breakout_long = trending_market and volume_spike and close[i] >= highest_high[i]
        breakout_short = trending_market and volume_spike and close[i] <= lowest_low[i]
        
        # Mean reversion signals (favor in ranging markets)
        donchian_mid = (highest_high[i] + lowest_low[i]) / 2
        mean_rev_long = ranging_market and volume_spike and close[i] <= lowest_low[i] and close[i] < donchian_mid
        mean_rev_short = ranging_market and volume_spike and close[i] >= highest_high[i] and close[i] > donchian_mid
        
        # Entry conditions
        long_entry = breakout_long or mean_rev_long
        short_entry = breakout_short or mean_rev_short
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals