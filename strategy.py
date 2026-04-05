#!/usr/bin/env python3
"""
exp_7503_4h_donchian20_12h_ema_vol_v1
Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation.
In bull markets (price > 12h EMA200): buy breakout above Donchian(20) high with volume spike.
In bear markets (price < 12h EMA200): sell breakout below Donchian(20) low with volume spike.
Targets 75-200 total trades over 4 years (19-50/year) with strict breakout + volume conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7503_4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_TREND = 200
VOLUME_MA = 20
VOLUME_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA200 for trend filter
    close_12h = df_12h['close'].values
    ema_12h_200 = pd.Series(close_12h).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_12h_200_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_200)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume filter
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA, min_periods=VOLUME_MA).mean().values
    
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
    start = max(DONCHIAN_PERIOD, EMA_TREND, VOLUME_MA, ATR_PERIOD)
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_12h_200_aligned[i]):
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
        
        # Determine market regime
        above_ema200 = close[i] > ema_12h_200_aligned[i]  # bull regime
        below_ema200 = close[i] < ema_12h_200_aligned[i]  # bear regime
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # break above previous period's high
        breakout_down = close[i] < lowest_low[i-1]  # break below previous period's low
        
        # Volume confirmation
        volume_spike = volume[i] > VOLUME_MULTIPLIER * volume_ma[i]
        
        # Entry conditions
        long_entry = (
            above_ema200 and           # bull regime
            breakout_up and            # Donchian breakout up
            volume_spike               # volume confirmation
        )
        
        short_entry = (
            below_ema200 and           # bear regime
            breakout_down and          # Donchian breakout down
            volume_spike               # volume confirmation
        )
        
        # Exit conditions
        long_exit = close[i] < lowest_low[i]  # exit when price breaks below Donchian low
        short_exit = close[i] > highest_high[i]  # exit when price breaks above Donchian high
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals