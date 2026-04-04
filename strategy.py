#!/usr/bin/env python3
"""
exp_6497_4h_donchian20_1d_ema_vol_v2
Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
Uses tighter volume threshold (2.2x) and adds ATR-based stoploss to reduce whipsaws.
Designed for fewer, higher-quality trades (target: 50-100 total over 4 years) to overcome fee drift.
Uses 4h primary timeframe per experiment instructions.
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6497_4h_donchian20_1d_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 2.2  # volume must be 2.2x its 20-period MA (tighter)
ATR_PERIOD = 14
ATR_STOP_MULT = 2.5  # stoploss at 2.5x ATR
SIGNAL_SIZE = 0.25   # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, min_periods=EMA_PERIOD, adjust=False).mean().values
    
    # Align to LTF (4h) with shift(1) for completed bars only
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
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
    tr1 = pd.Series(high).rolling(2).apply(lambda x: x[1] - x[0], raw=True).abs().values
    tr2 = pd.Series(high).rolling(2).apply(lambda x: abs(x[1] - close[int(x.index[0])] if len(x)==2 else 0), raw=True).values
    tr3 = pd.Series(low).rolling(2).apply(lambda x: abs(close[int(x.index[0])] if len(x)==2 else 0 - x[1]), raw=True).values
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Fix first element
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=ATR_PERIOD, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA or ATR data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]):
            continue
            
        # Long conditions: price breaks above Donchian HIGH + above 1d EMA + volume spike
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_trend = close[i] > ema_1d_aligned[i]  # price above 1d EMA (bullish trend)
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price breaks below Donchian LOW + below 1d EMA + volume spike
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_trend = close[i] < ema_1d_aligned[i]  # price below 1d EMA (bearish trend)
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Manage existing positions
        if position == 1:  # long position
            # Check stoploss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
            # Exit if price drops below midpoint of channel
            midpoint = (donchian_high[i-1] + donchian_low[i-1]) / 2
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
                continue
            # Otherwise hold
            signals[i] = SIGNAL_SIZE
            continue
            
        elif position == -1:  # short position
            # Check stoploss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
            # Exit if price rises above midpoint of channel
            midpoint = (donchian_high[i-1] + donchian_low[i-1]) / 2
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
                continue
            # Otherwise hold
            signals[i] = -SIGNAL_SIZE
            continue
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and long_trend and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - ATR_STOP_MULT * atr[i]
            elif short_breakout and short_trend and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + ATR_STOP_MULT * atr[i]
            else:
                signals[i] = 0.0
        else:
            # Should not reach here due to continue statements above
            signals[i] = position * SIGNAL_SIZE
    
    return signals