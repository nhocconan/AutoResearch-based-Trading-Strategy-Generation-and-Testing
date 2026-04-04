#!/usr/bin/env python3
"""
exp_6500_4h_donchian20_1d_ema_vol_v3
Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
Uses 1d EMA(50) as trend filter: long only when price > EMA50, short only when price < EMA50.
Donchian(20) breakout provides entry timing, volume confirmation filters weak breakouts.
ATR-based stoploss (2*ATR) limits downside. Designed for higher Sharpe via tighter risk control.
Target: 100-200 trades over 4 years (25-50/year) to balance statistical validity and fee drag.
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6500_4h_donchian20_1d_ema_vol_v3"
timeframe = "4h"
leverage = 1.0

# Parameters - optimized for better risk-adjusted returns
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 1.8  # Increased to reduce false signals
SIGNAL_SIZE = 0.25   # 25% position size
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0  # Stoploss at 2*ATR

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
        # Skip if EMA or ATR data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]):
            continue
            
        # Check stoploss for existing positions
        if position == 1:  # long position
            if close[i] < entry_price - ATR_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] > entry_price + ATR_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Long conditions: price breaks above Donchian HIGH + above 1d EMA + volume spike
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_trend = close[i] > ema_1d_aligned[i]  # price above 1d EMA (bullish trend)
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price breaks below Donchian LOW + below 1d EMA + volume spike
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_trend = close[i] < ema_1d_aligned[i]  # price below 1d EMA (bearish trend)
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and long_trend and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_breakout and short_trend and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals