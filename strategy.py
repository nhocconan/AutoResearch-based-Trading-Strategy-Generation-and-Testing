#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In strong trends, 
# extreme readings can signal continuation rather than reversal.
# Trend filter: 1d EMA(50) slope determines market regime.
# Entry: Williams %R < -80 in bull trend (oversold in uptrend = buy dip) 
#        or Williams %R > -20 in bear trend (overbought in downtrend = sell rally).
# Exit: Opposite signal or stop loss at 2*ATR.
# Works in bull markets (buying dips) and bear markets (selling rallies).

name = "exp_13591_6h_williamsr_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
WILLIAMS_PERIOD = 14
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr.values

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
    
    # Load 1d data for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend regime
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    ema_1d_slope = np.diff(ema_1d, prepend=ema_1d[0])  # slope approximation
    ema_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_slope)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R
    williams_r = calculate_williams_r(high, low, close, WILLIAMS_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WILLIAMS_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_slope_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(volume_ma[i]):
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
        
        # Trend regime from 1d EMA slope
        bull_trend = ema_1d_slope_aligned[i] > 0
        bear_trend = ema_1d_slope_aligned[i] < 0
        
        # Williams %R signals with trend filter
        # In bull trend: buy when oversold (%R < -80)
        # In bear trend: sell when overbought (%R > -20)
        long_signal = volume_ok and williams_r[i] < -80 and bull_trend
        short_signal = volume_ok and williams_r[i] > -20 and bear_trend
        
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
            # Exit long when overbought in bull trend or trend turns bearish
            if williams_r[i] > -20 or bear_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when oversold in bear trend or trend turns bullish
            if williams_r[i] < -80 or bull_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals