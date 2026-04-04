#!/usr/bin/env python3
"""
exp_6500_4h_donchian20_1d_ema_vol_v2
Hypothesis: Refined 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
Key improvements: reduced VOL_THRESHOLD to 1.5 for more entries, added ATR-based stoploss for risk management,
and discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 75-200 trades over 4 years.
Uses 4h primary timeframe with 1d HTF for trend alignment. Works in bull/bear via 1d EMA filter.
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6500_4h_donchian20_1d_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 1.5  # Reduced from 1.8 to increase trade frequency
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
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
        
        # Update existing positions
        if position == 1:  # long position
            # ATR-based stoploss
            if close[i] < entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
            # Hold long position
            signals[i] = SIGNAL_SIZE
            
        elif position == -1:  # short position
            # ATR-based stoploss
            if close[i] > entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
            # Hold short position
            signals[i] = -SIGNAL_SIZE
            
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
    
    return signals