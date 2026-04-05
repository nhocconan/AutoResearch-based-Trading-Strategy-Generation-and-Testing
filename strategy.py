#!/usr/bin/env python3
"""
Experiment #8534: 1h strategy with 4h/1d trend filter + volume confirmation + mean reversion entry.
Hypothesis: In ranging markets (2025 test period), price reverts to 20-period mean with volume confirmation.
Uses 4h for trend direction (avoid counter-trend) and 1d for volatility filter (avoid chop).
Targets 100-200 total trades over 4 years (25-50/year) to balance frequency and edge.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8534_1h_meanrev_4h_trend_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
MEAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
TREND_PERIOD = 50
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=TREND_PERIOD, adjust=False, min_periods=TREND_PERIOD).mean().values
    # Price relative to 4h EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_4h > ema_4h, 1, 
                     np.where(close_4h < ema_4h, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    trend_4h = align_htf_to_ltf(prices, df_4h, price_vs_ema)
    
    # Calculate 1d ATR for volatility filter (avoid high volatility periods)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    # Use 20-period SMA of ATR as volatility baseline
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    vol_filter_1d = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Price relative to 20-period mean (mean reversion target)
    price_ma = pd.Series(close).rolling(window=MEAN_PERIOD, min_periods=MEAN_PERIOD).mean().values
    price_dev = (close - price_ma) / price_ma  # Deviation from mean
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC (reduce noise outside active hours)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(MEAN_PERIOD, VOLUME_MA_PERIOD, TREND_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_4h[i]) or np.isnan(vol_filter_1d[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check session: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE  # hold position
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
        
        # Volatility filter: avoid trading when volatility is too high (>1.5x average)
        vol_condition = vol_filter_1d[i] > 0 and atr[i] < (1.5 * vol_filter_1d[i])
        
        # Mean reversion conditions
        # Long when price is significantly below mean AND volume confirms
        long_entry = (price_dev[i] < -0.015) and volume_confirmed and vol_condition and (trend_4h[i] != -1)
        # Short when price is significantly above mean AND volume confirms
        short_entry = (price_dev[i] > 0.015) and volume_confirmed and vol_condition and (trend_4h[i] != 1)
        
        # Volume confirmation (using previous bar to avoid look-ahead)
        volume_confirmed = (i > 0 and not np.isnan(volume_ma[i-1]) and 
                          volume[i-1] > (volume_ma[i-1] * VOLUME_THRESHOLD))
        
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