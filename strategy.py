#!/usr/bin/env python3
"""
Experiment #8254: 1-hour RSI reversal with 4h trend filter and volume confirmation.
Hypothesis: In ranging markets, RSI extremes on 1h with volume spike and aligned 4h trend (price above/below 4h EMA50) provide high-probability reversals. The 4h trend filter ensures we trade with the higher timeframe momentum, reducing false signals in chop. Target 60-150 trades over 4 years by using strict RSI thresholds (<20 for long, >80 for short) and volume confirmation.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8254_1h_rsi20_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 80
RSI_OVERSOLD = 20
EMA_PERIOD = 50
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    price_above_ema = close_4h > ema_4h  # True = bullish trend
    price_above_ema_aligned = align_htf_to_ltf(prices, df_4h, price_above_ema)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection
    volume_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * VOLUME_SPIKE_MULTIPLIER)
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(np.roll(close, 1) - high)
    tr3 = np.abs(np.roll(close, 1) - low)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_above_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Determine trend bias from 4h EMA
        bull_trend = price_above_ema_aligned[i]   # True if 4h close above EMA50
        bear_trend = not price_above_ema_aligned[i]  # True if 4h close below EMA50
        
        # Entry conditions: RSI extremes with volume spike and trend alignment
        long_signal = (rsi[i] < RSI_OVERSOLD) and volume_spike[i] and bull_trend
        short_signal = (rsi[i] > RSI_OVERBOUGHT) and volume_spike[i] and bear_trend
        
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
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals