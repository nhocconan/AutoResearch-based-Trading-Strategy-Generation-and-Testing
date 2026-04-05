#!/usr/bin/env python3
"""
Experiment #7614: 1h with 4h/1d trend filter - targeting 15-37 trades/year
Hypothesis: Use 4h Donchian(20) for signal direction and 1d EMA200 for regime filter.
Only take long when price > 1d EMA200 and break above 4h Donchian upper.
Only take short when price < 1d EMA200 and break below 4h Donchian lower.
Volume must be > 1.5x 20-period average to confirm breakout.
Trade only during 08-20 UTC session to reduce noise.
Position size fixed at 0.20 to manage drawdown.
Target: 60-150 total trades over 4 years (15-37/year) with strict entry conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7614_1h_4h_1d_trend_filter_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20          # 4h Donchian period
EMA_TREND = 200               # 1d EMA for trend filter
VOLUME_MA_PERIOD = 20         # Volume moving average
VOLUME_THRESHOLD = 1.5        # Volume must be 1.5x average
SIGNAL_SIZE = 0.20            # Fixed position size (20%)
ATR_PERIOD = 14               # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.5     # Stop loss at 2.5x ATR

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Donchian channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    highest_high = pd.Series(high_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    donchian_high = align_htf_to_ltf(prices, df_4h, highest_high)
    donchian_low = align_htf_to_ltf(prices, df_4h, lowest_low)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_1d_200 = pd.Series(close_1d).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_TREND, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_1d_200_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Only trade during session
        if not in_session:
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Determine market regime
        bull_regime = close[i] > ema_1d_200_aligned[i]   # price above 1d EMA200
        bear_regime = close[i] < ema_1d_200_aligned[i]   # price below 1d EMA200
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions (using previous bar's Donchian levels)
        upper_breakout = (high[i] > donchian_high[i-1]) and (i-1 >= 0) and not np.isnan(donchian_high[i-1])
        lower_breakout = (low[i] < donchian_low[i-1]) and (i-1 >= 0) and not np.isnan(donchian_low[i-1])
        
        # Entry conditions
        long_entry = bull_regime and upper_breakout and volume_confirmed
        short_entry = bear_regime and lower_breakout and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals