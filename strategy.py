#!/usr/bin/env python3
"""
exp_6525_12h_donchian20_1d_ema_vol_v1
Hypothesis: 12h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation.
In bull markets (price > 1d EMA200): long when price breaks above Donchian high with volume > 2.0x MA.
In bear markets (price < 1d EMA200): short when price breaks below Donchian low with volume > 2.0x MA.
Uses discrete position sizing (0.25) to minimize fee churn and ATR-based stoploss (2.5*ATR).
Designed for low-frequency, high-conviction trades targeting 75-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6525_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 200
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 2.0  # volume must be 2.0x its 20-period MA for confirmation
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.5  # stoploss at 2.5 * ATR
SIGNAL_SIZE = 0.25   # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 250:  # need enough data for EMA200 and ATR
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA200
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False).mean().values
    
    # Calculate 1d ATR for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first period has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    # Align to LTF (12h) with shift(1) for completed bars only
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]):
            continue
            
        # Check stoploss for existing positions
        if position == 1:  # long position
            # Stoploss: price drops below entry_price - ATR_MULTIPLIER * atr_1d_aligned[i]
            if close[i] < entry_price - ATR_MULTIPLIER * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Stoploss: price rises above entry_price + ATR_MULTIPLIER * atr_1d_aligned[i]
            if close[i] > entry_price + ATR_MULTIPLIER * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Long conditions: price > 1d EMA200 (bullish bias) + breaks above Donchian HIGH + volume spike
        long_bias = close[i] > ema_1d_aligned[i]  # price above 1d EMA200 (bullish)
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price < 1d EMA200 (bearish bias) + breaks below Donchian LOW + volume spike
        short_bias = close[i] < ema_1d_aligned[i]  # price below 1d EMA200 (bearish)
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Enter new positions only if flat
        if position == 0:
            if long_bias and long_breakout and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_bias and short_breakout and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals