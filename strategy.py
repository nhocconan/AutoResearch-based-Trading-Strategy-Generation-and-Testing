#!/usr/bin/env python3
"""
Experiment #8302: 12-hour Camarilla pivot with 1-day volume spike and choppiness regime filter.
Hypothesis: Price touching Camarilla pivot levels (L3/H3) on 12h with volume >2x 24-period MA 
and choppiness index >61.8 (range regime) captures mean-reversion bounces in both bull and bear markets. 
The choppiness filter avoids trending markets where mean reversion fails, while volume confirms 
institutional interest at pivot levels. Targeting 50-150 total trades over 4 years for optimal balance.
"""

from mtf_data import get_ldt_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8302_12h_camarilla1d_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1
VOLUME_MA_PERIOD = 24
VOLUME_THRESHOLD = 2.0
CHOPPINESS_PERIOD = 14
CHOPPINESS_THRESHOLD = 61.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (using previous day's OHLC)
    # H3 = Close + (High - Low) * 1.1/4
    # L3 = Close - (High - Low) * 1.1/4
    rng = high_1d - low_1d
    camarilla_h3 = close_1d + (rng * 1.1 / 4)
    camarilla_l3 = close_1d - (rng * 1.1 / 4)
    
    # Calculate Chopiness Index
    def calculate_chop(high, low, close, period):
        atr = []
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr_val = np.zeros_like(close)
        atr_val[0] = tr[0]
        for i in range(1, len(close)):
            atr_val[i] = (atr_val[i-1] * (period-1) + tr[i]) / period
        
        # Sum of true ranges over period
        tr_sum = np.convolve(tr, np.ones(period), 'full')[period-1:len(tr)+period-1]
        # Absolute sum of net change
        net_change = np.abs(np.diff(close, prepend=close[0]))
        nc_sum = np.convolve(net_change, np.ones(period), 'full')[period-1:len(net_change)+period-1]
        
        # Chop = 100 * log10(nc_sum/tr_sum) / log10(period)
        chop = 100 * np.log10(nc_sum / tr_sum) / np.log10(period)
        return chop
    
    chop = calculate_chop(high_1d, low_1d, close_1d, CHOPPINESS_PERIOD)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Align HTF data to LTF
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, CHOPPINESS_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(chop_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]):
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
        
        # Range regime: chop > 61.8
        range_regime = chop_aligned[i] > CHOPPINESS_THRESHOLD
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Price near Camarilla levels (within 0.1% tolerance)
        near_h3 = np.abs(close[i] - camarilla_h3_aligned[i]) / camarilla_h3_aligned[i] < 0.001
        near_l3 = np.abs(close[i] - camarilla_l3_aligned[i]) / camarilla_l3_aligned[i] < 0.001
        
        # Entry conditions: mean reversion at extremes in range
        long_entry = range_regime and volume_confirmed and near_l3
        short_entry = range_regime and volume_confirmed and near_h3
        
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