#!/usr/bin/env python3
"""
exp_6683_4h_donchian20_1d_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with 1-day EMA trend filter and volume confirmation.
In bull markets: buy when price breaks above 20-period Donchian high with 1d EMA up and volume spike.
In bear markets: sell when price breaks below 20-period Donchian low with 1d EMA down and volume spike.
ATR-based stoploss to limit drawdown. Designed for 4h timeframe to capture medium-term trends
while minimizing fee drag (~20-50 trades/year expected). Uses discrete position sizing (0.25) to
reduce churn and works in both bull and bear regimes by following the 1d EMA trend.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6683_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50  # 1-day EMA for trend filter
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
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
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOL_MA_PERIOD, ATR_PERIOD)
    
    for i in range(start, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD
        
        # Trend filter: 1-day EMA direction
        # For first bar, compare with previous day's EMA
        if i > 0 and not np.isnan(ema_1d_aligned[i-1]):
            ema_uptrend = ema_1d_aligned[i] > ema_1d_aligned[i-1]
            ema_downtrend = ema_1d_aligned[i] < ema_1d_aligned[i-1]
        else:
            ema_uptrend = False
            ema_downtrend = False
        
        # Breakout conditions
        bull_breakout = (close[i] > donchian_high[i-1]) and ema_uptrend and vol_confirmed
        bear_breakout = (close[i] < donchian_low[i-1]) and ema_downtrend and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if bull_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif bear_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals