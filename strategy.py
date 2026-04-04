#!/usr/bin/env python3
"""
exp_6533_4h_donchian20_12h_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 as trend filter and volume confirmation.
Uses 12h HTF for trend (more responsive than 1d) to capture medium-term shifts.
Volume > 1.5x 20-period MA confirms breakout strength.
ATR-based stoploss: exit when price moves 2.5*ATR against position.
Designed for 75-200 total trades over 4 years with discrete sizing to minimize fee churn.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6533_4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50          # 12h EMA50 for medium-term trend
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 1.5      # volume must be 1.5x its MA for confirmation
SIGNAL_SIZE = 0.25       # 25% position size
ATR_PERIOD = 14
ATR_MULT = 2.5           # ATR multiplier for stoploss

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for EMA50
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_PERIOD, adjust=False).mean().values
    
    # Align to LTF (4h) with shift(1) for completed bars only
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
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
    
    # ATR for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=ATR_PERIOD, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_12h_aligned[i]) or np.isnan(atr[i]):
            continue
            
        # Long conditions: price > 12h EMA50 (bullish bias) + breaks above Donchian HIGH + volume confirmation
        long_bias = close[i] > ema_12h_aligned[i]  # price above 12h EMA50
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price < 12h EMA50 (bearish bias) + breaks below Donchian LOW + volume confirmation
        short_bias = close[i] < ema_12h_aligned[i]  # price below 12h EMA50
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Update position and check stoploss
        if position == 1:  # long position
            # Check stoploss: price dropped 2.5*ATR below entry
            if close[i] <= entry_price - ATR_MULT * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
            # Hold long position
            signals[i] = SIGNAL_SIZE
            continue
        elif position == -1:  # short position
            # Check stoploss: price rose 2.5*ATR above entry
            if close[i] >= entry_price + ATR_MULT * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
            # Hold short position
            signals[i] = -SIGNAL_SIZE
            continue
        
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
            # Should not reach here due to continue statements above
            signals[i] = position * SIGNAL_SIZE
    
    return signals