#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h volume-weighted mean reversion with 12h trend filter.
# Uses volume-weighted average price (VWAP) as dynamic mean.
# In strong uptrend (12h EMA > VWAP): buy when price crosses below VWAP (mean reversion).
# In strong downtrend (12h EMA < VWAP): sell when price crosses above VWAP (mean reversion).
# Volume filter ensures high participation during reversals.
# Works in bull markets (buy dips) and bear markets (sell rallies).

name = "exp_13599_6h_vwap_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
VWAP_PERIOD = 20
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_vwap(high, low, close, volume, period):
    """Calculate Volume Weighted Average Price (VWAP)"""
    typical_price = (high + low + close) / 3
    vwap_numerator = pd.Series(typical_price * volume).rolling(window=period, min_periods=period).sum()
    vwap_denominator = pd.Series(volume).rolling(window=period, min_periods=period).sum()
    vwap = vwap_numerator / vwap_denominator
    vwap = vwap.fillna(method='bfill').fillna(method='ffill').fillna(0).values
    return vwap

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
    
    # Load 12h data for trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP
    vwap = calculate_vwap(high, low, close, volume, VWAP_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VWAP_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_12h_aligned[i]) or np.isnan(vwap[i]) or np.isnan(volume_ma[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 12h EMA vs VWAP
        price_above_ema = close[i] > ema_12h_aligned[i]
        price_below_ema = close[i] < ema_12h_aligned[i]
        
        # VWAP signals: cross above/below VWAP
        if i > 0 and not np.isnan(vwap[i-1]):
            vwap_prev = vwap[i-1]
            vwap_curr = vwap[i]
            
            # Long signal: price crosses above VWAP in uptrend (price > EMA)
            long_signal = volume_ok and price_above_ema and vwap_prev >= close[i-1] and vwap_curr < close[i]
            
            # Short signal: price crosses below VWAP in downtrend (price < EMA)
            short_signal = volume_ok and price_below_ema and vwap_prev <= close[i-1] and vwap_curr > close[i]
        else:
            long_signal = False
            short_signal = False
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on opposite VWAP signal or stop loss
            if i > 0 and not np.isnan(vwap[i-1]):
                vwap_prev = vwap[i-1]
                vwap_curr = vwap[i]
                # Exit if price crosses below VWAP (loss of mean reversion)
                if vwap_prev < close[i-1] and vwap_curr > close[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = SIGNAL_SIZE
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite VWAP signal or stop loss
            if i > 0 and not np.isnan(vwap[i-1]):
                vwap_prev = vwap[i-1]
                vwap_curr = vwap[i]
                # Exit if price crosses above VWAP (loss of mean reversion)
                if vwap_prev > close[i-1] and vwap_curr < close[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals