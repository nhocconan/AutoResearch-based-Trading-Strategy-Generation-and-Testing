#!/usr/bin/env python3
"""
exp_6514_1h_donchian20_4h_ema1d_vol_v1
Hypothesis: 1h Donchian(20) breakout with 4h EMA trend filter and 1d EMA regime filter, volume confirmation.
Uses 4h EMA(50) for intermediate trend direction and 1d EMA(200) for bull/bear regime.
Donchian(20) breakout provides entry timing in the direction of 4h EMA trend, only when aligned with 1d EMA regime.
Volume confirmation filters weak breakouts. Session filter (08-20 UTC) reduces noise trades.
Designed to work in both bull and bear markets by requiring alignment between 4h trend and 1d regime.
Target: 60-150 total trades over 4 years = 15-37/year for 1h.
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6514_1h_donchian20_4h_ema1d_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_4H_PERIOD = 50
EMA_1D_PERIOD = 200
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 1.8  # volume must be 1.8x its 20-period MA
SIGNAL_SIZE = 0.20   # 20% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_4H_PERIOD, min_periods=EMA_4H_PERIOD, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d EMA(200) for regime filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_1D_PERIOD, min_periods=EMA_1D_PERIOD, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_4H_PERIOD, EMA_1D_PERIOD, VOL_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend and regime
        uptrend_4h = close[i] > ema_4h_aligned[i]  # price above 4h EMA = bullish trend
        downtrend_4h = close[i] < ema_4h_aligned[i]  # price below 4h EMA = bearish trend
        bull_regime = close[i] > ema_1d_aligned[i]   # price above 1d EMA = bull regime
        bear_regime = close[i] < ema_1d_aligned[i]   # price below 1d EMA = bear regime
        
        # Long conditions: price breaks above Donchian HIGH + 4h uptrend + bull regime + volume spike
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price breaks below Donchian LOW + 4h downtrend + bear regime + volume spike
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: opposite 4h EMA cross
        if position == 1:  # long position
            # Exit if price drops below 4h EMA (trend change)
            exit_long = close[i] < ema_4h_aligned[i]
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises above 4h EMA (trend change)
            exit_short = close[i] > ema_4h_aligned[i]
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and uptrend_4h and bull_regime and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_breakout and downtrend_4h and bear_regime and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals