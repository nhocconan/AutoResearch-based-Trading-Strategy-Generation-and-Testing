#!/usr/bin/env python3
"""
exp_6483_4h_donchian20_12h_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation.
Enters long when price breaks above Donchian high, 12h EMA is rising, and volume spikes.
Enters short when price breaks below Donchian low, 12h EMA is falling, and volume spikes.
Uses ATR-based stoploss to limit drawdown. Designed for 4h timeframe with target 75-200 trades over 4 years.
Uses daily pivot points as structural bias filter: long only when price > daily pivot, short only when price < daily pivot.
Combines trend (EMA), momentum (Donchian breakout), structure (pivot), and volume confirmation for robustness.
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6483_4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 21
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 2.0  # volume must be 2.0x its 20-period MA
PIVOT_SOURCE = '1d'  # daily pivot for structural bias
SIGNAL_SIZE = 0.25   # 25% position size
ATR_PERIOD = 14
ATR_STOP_MULT = 2.5  # stoploss at 2.5 * ATR

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for EMA and 1d for pivot
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_PERIOD, min_periods=EMA_PERIOD, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate daily pivot points for structural bias
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot point: (high + low + close) / 3
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    # Align to LTF (4h) with shift(1) for completed bars only
    daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    
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
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if pivot or EMA data not available
        if np.isnan(daily_pivot_aligned[i]) or np.isnan(ema_12h_aligned[i]):
            continue
            
        # Check stoploss for existing position
        if position == 1:  # long position
            if close[i] < entry_price - ATR_STOP_MULT * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] > entry_price + ATR_STOP_MULT * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Long conditions: price breaks above Donchian HIGH + EMA rising + above daily pivot + volume spike
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        ema_rising = ema_12h_aligned[i] > ema_12h_aligned[i-1]  # 12h EMA rising
        long_bias = close[i] > daily_pivot_aligned[i]  # price above daily pivot (bullish bias)
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price breaks below Donchian LOW + EMA falling + below daily pivot + volume spike
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        ema_falling = ema_12h_aligned[i] < ema_12h_aligned[i-1]  # 12h EMA falling
        short_bias = close[i] < daily_pivot_aligned[i]  # price below daily pivot (bearish bias)
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and ema_rising and long_bias and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_breakout and ema_falling and short_bias and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals