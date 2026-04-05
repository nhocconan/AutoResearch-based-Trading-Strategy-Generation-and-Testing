#!/usr/bin/env python3
"""
Experiment #7679: 6-hour ADX/ATR Breakout with 12-hour ATR Regime Filter.
Hypothesis: Use ADX to detect trending markets and ATR-based breakouts for entry.
In trending regimes (ADX > 25), enter on ATR-based breakouts in the direction of the trend.
In ranging regimes (ADX < 20), fade at extreme ATR deviations from mean.
12-hour ATR regime filter prevents entries during extreme volatility expansions.
Targets 80-150 total trades over 4 years (20-38/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7679_6h_adxatr_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ATR_PERIOD = 14
ADX_PERIOD = 14
ATR_MA_PERIOD = 50
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
ATR_MULT_BREAKOUT = 1.5
ATR_MULT_FADE = 2.5
SIGNAL_SIZE = 0.28

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h ATR for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1_12h = pd.Series(high_12h - low_12h)
    tr2_12h = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3_12h = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h = tr_12h.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    atr_ma_12h = pd.Series(atr_12h).ewm(span=ATR_MA_PERIOD, adjust=False, min_periods=ATR_MA_PERIOD).mean().values
    atr_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ma_12h)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range and ATR
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # ADX calculation
    plus_dm = pd.Series(np.where((high - high.shift(1)) > (low.shift(1) - low), 
                                 np.maximum(high - high.shift(1), 0), 0))
    minus_dm = pd.Series(np.where((low.shift(1) - low) > (high - high.shift(1)), 
                                  np.maximum(low.shift(1) - low, 0), 0))
    
    tr_for_adx = tr.copy()
    atr_for_adx = tr_for_adx.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    plus_di = 100 * (plus_dm.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_for_adx)
    minus_di = 100 * (minus_dm.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_for_adx)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ATR_PERIOD * 2, ADX_PERIOD * 2, ATR_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(atr_ma_12h_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Volatility regime filter: avoid extreme ATR expansion
        vol_regime_ok = atr[i] < (atr_ma_12h_aligned[i] * 2.0)
        
        # Skip if ATR data not ready
        if np.isnan(atr[i]) or np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Check stoploss (2x ATR)
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
        trending = adx[i] > ADX_TREND_THRESHOLD
        ranging = adx[i] < ADX_RANGE_THRESHOLD
        
        # Calculate ATR-based bands
        atr_mult = ATR_MULT_BREAKOUT if trending else ATR_MULT_FADE
        upper_band = close[i-1] + (atr[i] * atr_mult) if i > 0 else close[i]
        lower_band = close[i-1] - (atr[i] * atr_mult) if i > 0 else close[i]
        
        # Entry conditions
        long_entry = False
        short_entry = False
        
        if trending:
            # In trending markets: breakout in direction of DI crossover
            if plus_di[i] > minus_di[i] and high[i] > upper_band:
                long_entry = True
            elif minus_di[i] > plus_di[i] and low[i] < lower_band:
                short_entry = True
        elif ranging:
            # In ranging markets: fade at extreme deviations
            if high[i] > upper_band:
                short_entry = True  # fade the breakout
            elif low[i] < lower_band:
                long_entry = True   # fade the breakdown
        
        # Generate signals
        if position == 0:
            if long_entry and vol_regime_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_entry and vol_regime_ok:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals