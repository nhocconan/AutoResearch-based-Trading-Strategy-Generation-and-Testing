#!/usr/bin/env python3
"""
exp_7510_1d_1w_donchian20_ema_vol_v1
Hypothesis: 1d Donchian breakout with 1w EMA trend filter and volume confirmation. 
Breakouts above Donchian(20) high in bull market (price > 1w EMA50) go long.
Breakouts below Donchian(20) low in bear market (price < 1w EMA50) go short.
Volume must be > 1.5x average to confirm breakout strength.
Targets 30-100 trades over 4 years (7-25/year) with strong breakouts only.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7510_1d_1w_donchian20_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_TREND = 50
VOLUME_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    start = max(DONCHIAN_PERIOD, EMA_TREND, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_50_aligned[i]):
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
        bull_market = close[i] > ema_1w_50_aligned[i]   # bull regime
        bear_market = close[i] < ema_1w_50_aligned[i]   # bear regime
        
        # Volume confirmation
        vol_confirmed = volume[i] > VOLUME_MULTIPLIER * avg_volume[i] if not np.isnan(avg_volume[i]) else False
        
        # Entry conditions
        long_entry = (
            bull_market and           # bull regime
            close[i] > donchian_high[i] and  # breakout above Donchian high
            vol_confirmed             # volume confirmation
        )
        
        short_entry = (
            bear_market and           # bear regime
            close[i] < donchian_low[i] and   # breakdown below Donchian low
            vol_confirmed             # volume confirmation
        )
        
        # Exit conditions: reverse signal on opposite breakout
        long_exit = close[i] < donchian_low[i]  # break below Donchian low
        short_exit = close[i] > donchian_high[i]  # break above Donchian high
        
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
                signals[i] = -SIGNAL_SIZE  # reverse to short
                position = -1
                entry_price = close[i]
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = SIGNAL_SIZE   # reverse to long
                position = 1
                entry_price = close[i]
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals