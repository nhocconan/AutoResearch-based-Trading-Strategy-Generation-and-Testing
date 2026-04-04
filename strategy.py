#!/usr/bin/env python3
"""
exp_6580_4h_donchian20_1d_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with EMA50 trend filter and volume confirmation.
Primary timeframe 4h balances trade frequency (target: 75-200 total trades over 4 years).
EMA50 provides trend alignment to avoid counter-trend trades. Volume confirms breakout conviction.
Discrete sizing (0.25) minimizes fee churn. Works in both bull and bear markets by following
institutional price levels (Donchian) with trend filter preventing whipsaws.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6580_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0  # Volume threshold for confirmation
SIGNAL_SIZE = 0.25      # 25% position size
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.5    # ATR stoploss multiplier

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for higher timeframe context (not used in logic but available)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    entry_atr = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Check stoploss for existing positions
        if position == 1:  # long position
            if close[i] < entry_price - ATR_MULTIPLIER * entry_atr:
                signals[i] = 0.0
                position = 0
                continue
            # Hold long
            signals[i] = SIGNAL_SIZE
        elif position == -1:  # short position
            if close[i] > entry_price + ATR_MULTIPLIER * entry_atr:
                signals[i] = 0.0
                position = 0
                continue
            # Hold short
            signals[i] = -SIGNAL_SIZE
        else:
            # Flat - look for new entries
            # Skip if indicators not ready
            if (np.isnan(donchian_high[i-1]) or np.isnan(donchian_low[i-1]) or 
                np.isnan(ema50[i]) or np.isnan(vol_ma[i])):
                signals[i] = 0.0
                continue
                
            # Long conditions:
            # 1. Price breaks above Donchian HIGH (breakout)
            # 2. Price above EMA50 (uptrend filter)
            # 3. Volume confirmation (above average)
            long_breakout = close[i] > donchian_high[i-1]
            long_trend = close[i] > ema50[i]
            long_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD
            
            # Short conditions:
            # 1. Price breaks below Donchian LOW (breakdown)
            # 2. Price below EMA50 (downtrend filter)
            # 3. Volume confirmation
            short_breakout = close[i] < donchian_low[i-1]
            short_trend = close[i] < ema50[i]
            short_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD
            
            if long_breakout and long_trend and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                entry_atr = atr[i] if not np.isnan(atr[i]) else 0.0
            elif short_breakout and short_trend and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                entry_atr = atr[i] if not np.isnan(atr[i]) else 0.0
            else:
                signals[i] = 0.0
    
    return signals