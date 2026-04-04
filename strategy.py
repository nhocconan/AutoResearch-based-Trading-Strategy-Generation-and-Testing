#!/usr/bin/env python3
"""
exp_6740_4h_donchian20_1d_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with 1d EMA trend filter and volume confirmation.
In bull markets: breakout continuation above 1d EMA. In bear markets: fade at Donchian extremes when price < 1d EMA.
Volume confirms breakout legitimacy. Designed for 4h timeframe to capture medium-term swings with ~19-50 trades/year (75-200 total over 4 years).
Works in both bull and bear markets by adapting to daily EMA context - trend following in bull, mean reversion in bear.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6740_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Align to LTF (4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
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
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
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
                
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on 1d EMA
        # Bull market: price > 1d EMA -> trend following
        # Bear market: price < 1d EMA -> mean reversion
        bull_market = close[i] > ema_1d_aligned[i] if not np.isnan(ema_1d_aligned[i]) else False
        
        # Trend following signals (bull market)
        long_trend = bull_market and (close[i] > highest_high[i]) and vol_confirmed
        short_trend = bull_market and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Mean reversion signals (bear market)
        long_mean_revert = (not bull_market) and (close[i] < lowest_low[i]) and vol_confirmed
        short_mean_revert = (not bull_market) and (close[i] > highest_high[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_trend or long_mean_revert:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_trend or short_mean_revert:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals