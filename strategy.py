#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray index with 1-day regime filter for bull/bear adaptation.
# Bull regime (price > 1-day EMA200): go long on Bull Power > 0
# Bear regime (price < 1-day EMA200): go short on Bear Power < 0
# Uses volume confirmation to filter weak signals. Works in both trending and ranging markets.
# Target: 80-150 total trades over 4 years (20-38/year).

name = "exp_13567_6d_elder_ray_1d_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA_LONG_PERIOD = 200
EMA_SHORT_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily indicators for regime filter
    close_1d = df_1d['close'].values
    ema_long_1d = calculate_ema(close_1d, EMA_LONG_PERIOD)
    ema_long_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_long_1d)
    
    # Calculate 6-hour indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Short-term EMA for Elder Ray
    ema_short = calculate_ema(close, EMA_SHORT_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume filter
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_LONG_PERIOD, EMA_SHORT_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if data not ready
        if np.isnan(ema_long_1d_aligned[i]) or np.isnan(ema_short[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
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
        
        # Elder Ray components
        bull_power = high[i] - ema_short[i]  # High - EMA(short)
        bear_power = low[i] - ema_short[i]   # Low - EMA(short)
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Regime filter: price vs daily EMA200
        price_above_ema200 = close[i] > ema_long_1d_aligned[i]
        price_below_ema200 = close[i] < ema_long_1d_aligned[i]
        
        # Entry logic
        if position == 0:
            # Long in bull regime: Bull Power positive with volume
            if price_above_ema200 and bull_power > 0 and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short in bear regime: Bear Power negative with volume
            elif price_below_ema200 and bear_power < 0 and volume_ok:
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