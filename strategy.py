# 1h Momentum with 4h Trend Filter and ATR Stop
# Designed for both bull and bear markets:
# - Bull: rides momentum with trend filter
# - Bear: avoids false signals with strong trend filter and tight stops
# Target: 50-150 trades over 4 years (12-38/year) to avoid fee drag
# Uses momentum (ROC) + trend (EMA) + volatility filter (ATR-based volume)
# Stops: 2x ATR trailing stop

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12677_1h_momentum_4h_trend_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
ROC_PERIOD = 10          # Rate of change period
EMA_FAST = 12            # Fast EMA for momentum
EMA_SLOW = 26            # Slow EMA for momentum
TREND_EMA_PERIOD = 50    # 4h EMA for trend filter
VOLUME_MA_PERIOD = 20    # Volume moving average
VOLUME_THRESHOLD = 1.5   # Volume must be 1.5x average
SIGNAL_SIZE = 0.25       # Position size (25% of capital)
ATR_PERIOD = 14          # ATR period for stop loss
ATR_STOP_MULTIPLIER = 2.0 # ATR multiplier for stop

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_rma(close, period):
    """Calculate RMA (Wilder's smoothing)"""
    return pd.Series(close).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Set first TR to first period's average to avoid NaN
    tr[0] = np.mean(tr[1:period+1]) if len(tr) > period else np.mean(tr[1:]) if len(tr) > 1 else 0
    atr = calculate_rma(tr, period)
    return atr

def calculate_roc(close, period):
    """Calculate Rate of Change"""
    roc = np.full_like(close, np.nan, dtype=float)
    roc[period:] = (close[period:] - close[:-period]) / close[:-period] * 100
    return roc

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend filter
    ema_4h = calculate_ema(df_4h['close'].values, TREND_EMA_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Momentum indicators
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    macd = ema_fast - ema_slow
    roc = calculate_roc(close, ROC_PERIOD)
    
    # Volatility and volume
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_SLOW, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, ROC_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 4h EMA not available
        if np.isnan(ema_4h_aligned[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter (4h)
        uptrend_4h = close[i] > ema_4h_aligned[i]
        downtrend_4h = close[i] < ema_4h_aligned[i]
        
        # Momentum conditions
        # Long: positive MACD and positive ROC
        long_momentum = macd[i] > 0 and roc[i] > 0
        # Short: negative MACD and negative ROC
        short_momentum = macd[i] < 0 and roc[i] < 0
        
        # Entry conditions
        long_entry = volume_ok and uptrend_4h and long_momentum
        short_entry = volume_ok and downtrend_4h and short_momentum
        
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