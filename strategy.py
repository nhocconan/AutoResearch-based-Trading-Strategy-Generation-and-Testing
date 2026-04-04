#!/usr/bin/env python3
"""
exp_6524_1d_donchian20_1w_ema_vol_v1
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
In bull markets (price > 1w EMA50): long when price breaks above Donchian high with volume > 1.8x MA.
In bear markets (price < 1w EMA50): short when price breaks below Donchian low with volume > 1.8x MA.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 30-100 trades over 4 years.
Designed for low-frequency, high-conviction trades with ATR-based stoploss (signal→0 on 2*ATR adverse move).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6524_1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 1.8  # volume must be 1.8x its 20-period MA
SIGNAL_SIZE = 0.25   # 25% position size
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0 # stoploss at 2*ATR

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for EMA50
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=EMA_PERIOD, adjust=False).mean().values
    
    # Align to LTF (1d) with shift(1) for completed bars only
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
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
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=ATR_PERIOD, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(atr[i]):
            continue
            
        # Check stoploss first
        if position == 1:  # long position
            if close[i] < entry_price - ATR_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        elif position == -1:  # short position
            if close[i] > entry_price + ATR_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # Long conditions: price > 1w EMA50 (bullish bias) + breaks above Donchian high + volume spike
        long_bias = close[i] > ema_1w_aligned[i]  # price above 1w EMA50 (bullish)
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price < 1w EMA50 (bearish bias) + breaks below Donchian low + volume spike
        short_bias = close[i] < ema_1w_aligned[i]  # price below 1w EMA50 (bearish)
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