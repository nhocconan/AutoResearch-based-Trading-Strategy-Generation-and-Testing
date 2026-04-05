#!/usr/bin/env python3
"""
Experiment #8174: 1-hour timeframe with 4h/1d trend filter and volume confirmation.
Hypothesis: 1h breakouts aligned with 4h trend (price above/below EMA) and 1d regime (price above/below EMA) with volume confirmation capture sustained moves while reducing whipsaw. 1h provides timely entries, 4h/1d filters reduce false signals. Target 15-37 trades/year via tight entry conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8174_1h_4h_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_PERIOD_4H = 34
EMA_PERIOD_1D = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0
SESSION_START_HOUR = 8   # UTC
SESSION_END_HOUR = 20    # UTC

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_PERIOD_4H, adjust=False, min_periods=EMA_PERIOD_4H).mean().values
    price_vs_ema_4h = np.where(close_4h > ema_4h, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_4h_aligned = align_htf_to_ltf(prices, df_4h, price_vs_ema_4h)
    
    # Calculate 1d EMA for regime filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD_1D, adjust=False, min_periods=EMA_PERIOD_1D).mean().values
    price_vs_ema_1d = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_1d_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema_1d)
    
    # Precompute session hours
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD_4H, EMA_PERIOD_1D, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_4h_aligned[i]) or np.isnan(price_vs_ema_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
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
        
        # Determine market bias from 4h EMA and 1d regime
        bull_bias_4h = price_vs_ema_4h_aligned[i] == 1   # 4h close above EMA34
        bear_bias_4h = price_vs_ema_4h_aligned[i] == -1  # 4h close below EMA34
        bull_bias_1d = price_vs_ema_1d_aligned[i] == 1   # 1d close above EMA50
        bear_bias_1d = price_vs_ema_1d_aligned[i] == -1  # 1d close below EMA50
        
        # Require alignment: both 4h and 1d agree on direction
        bull_aligned = bull_bias_4h and bull_bias_1d
        bear_aligned = bear_bias_4h and bear_bias_1d
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: aligned trend + volume
        long_entry = bull_aligned and volume_confirmed
        short_entry = bear_aligned and volume_confirmed
        
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