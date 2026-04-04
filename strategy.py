#!/usr/bin/env python3
"""
exp_6694_1h_donchian20_4h_ema_vol_v1
Hypothesis: 1h Donchian channel breakout with 4h EMA trend filter and volume confirmation.
In bull markets: buy breakouts above 20-period high when 4h EMA21 > EMA50.
In bear markets: sell breakdowns below 20-period low when 4h EMA21 < EMA50.
Uses 4h for signal direction (trend filter), 1h only for entry timing precision.
Adds session filter (08-20 UTC) to avoid low-liquidity periods.
Target: 60-150 total trades over 4 years = 15-37/year.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6694_1h_donchian20_4h_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_FAST = 21
EMA_SLOW = 50
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 4h for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMAs for trend direction
    close_4h = df_4h['close'].values
    ema_fast_4h = pd.Series(close_4h).ewm(span=EMA_FAST, adjust=False).mean().values
    ema_slow_4h = pd.Series(close_4h).ewm(span=EMA_SLOW, adjust=False).mean().values
    
    # Align HTF EMAs to LTF (1h) with shift(1) for completed 4h bars only
    ema_fast_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_fast_4h)
    ema_slow_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_slow_4h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period high/low)
    high_roll = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    low_roll = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    # Session filter: 08-20 UTC (avoid low liquidity)
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_SLOW, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if not in trading session
        if not in_session[i]:
            # Force flat outside session
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if HTF data not available
        if (np.isnan(ema_fast_4h_aligned[i]) or np.isnan(ema_slow_4h_aligned[i]) or
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine 4h trend direction
        uptrend_4h = ema_fast_4h_aligned[i] > ema_slow_4h_aligned[i]
        downtrend_4h = ema_fast_4h_aligned[i] < ema_slow_4h_aligned[i]
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD
        
        # Donchian breakout/breakdown signals
        breakout_up = close[i] > high_roll[i-1]  # Use previous bar's high to avoid look-ahead
        breakdown_down = close[i] < low_roll[i-1]  # Use previous bar's low
        
        # Enter new positions only if flat
        if position == 0:
            # Long: 4h uptrend + volume + breakout above 20-period high
            if uptrend_4h and vol_confirmed and breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            # Short: 4h downtrend + volume + breakdown below 20-period low
            elif downtrend_4h and vol_confirmed and breakdown_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals