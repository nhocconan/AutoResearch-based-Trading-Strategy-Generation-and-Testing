#!/usr/bin/env python3
"""
exp_7434_1h_volatility_squeeze_rsi_v1
Hypothesis: 1h Bollinger Band squeeze + RSI mean reversion with 4h trend filter.
In low volatility (squeeze), price tends to mean revert. Uses 4h EMA for trend direction to avoid counter-trend trades.
Targets 80-120 trades over 4 years (20-30/year) with 0.20 position size.
Works in bull/bear by following 4h trend: long only when price > 4h EMA, short only when price < 4h EMA.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7434_1h_volatility_squeeze_rsi_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
BB_PERIOD = 20
BB_STD = 2.0
SQUEEZE_THRESHOLD = 0.03  # Bandwidth threshold for squeeze
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_TREND_PERIOD = 50
VOLUME_SPIKE_MULT = 2.0
VOLUME_MA_PERIOD = 20
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 4h for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands
    sma = pd.Series(close).rolling(window=BB_PERIOD, min_periods=BB_PERIOD).mean().values
    std = pd.Series(close).rolling(window=BB_PERIOD, min_periods=BB_PERIOD).std().values
    upper_band = sma + (BB_STD * std)
    lower_band = sma - (BB_STD * std)
    bandwidth = (upper_band - lower_band) / sma
    
    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_spike = volume > (vol_ma * VOLUME_SPIKE_MULT)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(np.roll(close, 1) - high)
    tr3 = np.abs(np.roll(close, 1) - low)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(BB_PERIOD, RSI_PERIOD, EMA_TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Squeeze condition: low volatility
        is_squeeze = bandwidth[i] < SQUEEZE_THRESHOLD
        
        # Mean reversion signals within squeeze
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        price_at_lower = close[i] <= lower_band[i] * 1.001  # Near lower band
        price_at_upper = close[i] >= upper_band[i] * 0.999  # Near upper band
        
        # Trend alignment from 4h EMA
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Entry logic: mean reversion in squeeze with trend filter
        if position == 0:
            # Long: oversold + near lower band + uptrend on 4h
            if rsi_oversold and price_at_lower and uptrend and volume_spike[i] and is_squeeze:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            # Short: overbought + near upper band + downtrend on 4h
            elif rsi_overbought and price_at_upper and downtrend and volume_spike[i] and is_squeeze:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold position
            signals[i] = position * SIGNAL_SIZE
    
    return signals