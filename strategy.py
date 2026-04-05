#!/usr/bin/env python3
"""
Experiment #8094: 1-hour RSI reversal with 4h trend filter and volume confirmation.
Hypothesis: In both bull and bear markets, RSI extremes (oversold/overbought) on 1h 
combined with 4h trend alignment (price above/below 4h EMA50) and volume spike 
provides high-probability mean-reversion entries. Using 4h for trend direction 
reduces whipsaw, while 1h timing captures swings. Target: 60-150 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8094_1h_rsi40_4h_ema50_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 60
RSI_OVERSOLD = 40
EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Price relative to EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_4h > ema_4h, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_4h, price_vs_ema)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI (14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - (ATR_STOP_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + (ATR_STOP_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 4h EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 4h close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 4h close below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # RSI conditions
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Entry conditions
        long_entry = bull_bias and rsi_oversold and volume_confirmed
        short_entry = bear_bias and rsi_overbought and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals