#!/usr/bin/env python3
"""
exp_7563_4h_donchian20_1d_ema_vol_v2
Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA200 trend filter and volume confirmation.
Tightened entry conditions to reduce trade frequency: require volume > 2.0x average (was 1.5x)
and add ADX(14) > 25 filter to ensure trending markets only. Targets 50-150 trades over 4 years.
In bull markets (price > 1d EMA200): long breakout above 4h Donchian upper.
In bear markets (price < 1d EMA200): short breakdown below 4h Donchian lower.
ATR-based stoploss (2x) and target (3x) for risk management.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7563_4h_donchian20_1d_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_TREND = 200
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0  # Increased from 1.5 to reduce trades
ADX_PERIOD = 14
ADX_THRESHOLD = 25
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_1d_200 = pd.Series(close_1d).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # ADX for trend strength filter
    plus_dm = pd.Series(np.where((high - high.shift(1)) > (low.shift(1) - low), np.maximum(high - high.shift(1), 0), 0))
    minus_dm = pd.Series(np.where((low.shift(1) - low) > (high - high.shift(1)), np.maximum(low.shift(1) - low, 0), 0))
    tr_adx = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_adx = tr_adx.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean()
    plus_di = 100 * (plus_dm.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean() / atr_adx)
    minus_di = 100 * (minus_dm.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean() / atr_adx)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_TREND, VOLUME_MA_PERIOD, ATR_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1d_200_aligned[i]) or np.isnan(adx[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market regime
        bull_regime = close[i] > ema_1d_200_aligned[i]   # price above 1d EMA200
        bear_regime = close[i] < ema_1d_200_aligned[i]   # price below 1d EMA200
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend strength filter
        strong_trend = adx[i] > ADX_THRESHOLD
        
        # Breakout conditions
        upper_breakout = (high[i] > highest_high[i-1]) and (i-1 >= 0) and not np.isnan(highest_high[i-1])
        lower_breakout = (low[i] < lowest_low[i-1]) and (i-1 >= 0) and not np.isnan(lowest_low[i-1])
        
        # Entry conditions
        long_entry = bull_regime and upper_breakout and volume_confirmed and strong_trend
        short_entry = bear_regime and lower_breakout and volume_confirmed and strong_trend
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals