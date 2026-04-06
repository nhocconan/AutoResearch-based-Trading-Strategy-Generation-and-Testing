#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum breakouts filtered by 4h RSI and daily trend (EMA50).
# Uses 4h RSI to avoid overextended moves and daily EMA50 for trend alignment.
# Entry only during active London/NY session (08-20 UTC) to reduce noise.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Works in bull markets (breakouts above resistance with bullish 4h RSI) and
# bear markets (breakdowns below support with bearish 4h RSI).

name = "exp_13354_1h_momentum_4hrsi_dailyema_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20
VOLUME_MULTIPLIER = 1.5
VOLUME_AVG_PERIOD = 20

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

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
    
    # Load 4h data ONCE before loop for RSI
    df_4h = get_htf_data(prices, '4h')
    # Load daily data ONCE before loop for EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h RSI for trend filter
    close_4h = df_4h['close'].values
    rsi_4h = calculate_rsi(close_4h, RSI_PERIOD)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume average
    volume_avg = pd.Series(volume).rolling(window=VOLUME_AVG_PERIOD, min_periods=VOLUME_AVG_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, EMA_PERIOD, VOLUME_AVG_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(volume_avg[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
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
        volume_ok = volume[i] > (volume_avg[i] * VOLUME_MULTIPLIER)
        
        # Trend filters
        rsi_bullish = rsi_4h_aligned[i] > 50  # 4h bullish
        rsi_bearish = rsi_4h_aligned[i] < 50  # 4h bearish
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        # Momentum conditions: price breaking recent high/low
        lookback = 3
        recent_high = np.max(high[i-lookback:i])
        recent_low = np.min(low[i-lookback:i])
        breakout_up = close[i] > recent_high
        breakout_down = close[i] < recent_low
        
        # Generate signals only during session
        if in_session and volume_ok:
            if position == 0:
                # Long: 4h bullish, price above daily EMA, breaking recent high
                if rsi_bullish and price_above_ema and breakout_up:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                # Short: 4h bearish, price below daily EMA, breaking recent low
                elif rsi_bearish and price_below_ema and breakout_down:
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
        else:
            # Outside session or low volume: hold current position
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
    
    return signals