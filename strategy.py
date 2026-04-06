#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour RSI divergence with volume confirmation and 12-hour EMA trend filter.
# RSI detects overbought/oversold conditions; divergence with price signals potential reversals.
# Volume confirmation ensures institutional participation. 12-hour EMA filters counter-trend trades.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 80-180 total trades over 4 years (20-45/year).

name = "exp_13383_4h_rsi_divergence_vol_ema_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_PERIOD = 20  # 12-hour EMA
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
LOOKBACK_DIVERGENCE = 5  # bars to look back for divergence

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    
    # Load 12-hour data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12-hour EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4-hour indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, LOOKBACK_DIVERGENCE) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(rsi[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
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
        
        # Trend filter: price above/below 12h EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # RSI divergence detection
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        bearish_div = False
        
        if i >= LOOKBACK_DIVERGENCE:
            # Look back for divergence
            price_low = np.min(low[i-LOOKBACK_DIVERGENCE:i+1])
            price_high = np.max(high[i-LOOKBACK_DIVERGENCE:i+1])
            rsi_low = np.min(rsi[i-LOOKBACK_DIVERGENCE:i+1])
            rsi_high = np.max(rsi[i-LOOKBACK_DIVERGENCE:i+1])
            
            # Current price vs lookback period
            if low[i] < price_low and rsi[i] > rsi_low:
                bullish_div = True
            if high[i] > price_high and rsi[i] < rsi_high:
                bearish_div = True
        
        # Entry signals
        if position == 0:
            # Long: bullish divergence + oversold RSI + uptrend + volume
            if bullish_div and rsi[i] < RSI_OVERSOLD and uptrend and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: bearish divergence + overbought RSI + downtrend + volume
            elif bearish_div and rsi[i] > RSI_OVERBOUGHT and downtrend and volume_ok:
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