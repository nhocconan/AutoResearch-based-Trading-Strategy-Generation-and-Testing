#!/usr/bin/env python3
"""
Experiment #9274: 1h RSI(2) mean reversion + 4h trend filter + volume confirmation + 1d volatility filter.
Hypothesis: RSI(2) captures short-term reversals; 4h EMA50 ensures directional alignment; volume confirms momentum; 1d ATR filter avoids choppy markets. Targets 60-150 total trades over 4 years (15-37/year) to minimize fee drag. Works in bull (buy pullbacks in uptrend) and bear (sell bounces in downtrend).
"""

import numpy as np
import pandas as pd

name = "exp_9274_1h_rsi2_4h_trend_1d_vol_filter_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 2
RSI_OVERBOUGHT = 90
RSI_OVERSOLD = 10
EMA_TREND_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
ATR_PERIOD = 14
ATR_MA_PERIOD = 50
ATR_THRESHOLD = 0.5  # ATR must be above 50% of its MA to avoid chop
SIGNAL_SIZE = 0.20

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    ema_4h = pd.Series(close_4h).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    
    # Price relative to 4h EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_4h > ema_4h, 1, 
                     np.where(close_4h < ema_4h, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_4h, price_vs_ema)
    
    # Calculate 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=ATR_MA_PERIOD, min_periods=ATR_MA_PERIOD).mean().values
    atr_ratio = atr_1d / (atr_ma_1d + 1e-10)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(2)
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, EMA_TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, ATR_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(atr_ratio_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volatility filter: only trade when volatility is sufficient (not choppy)
        volatile_enough = atr_ratio_aligned[i] > ATR_THRESHOLD
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # RSI conditions
        rsi_overbought = rsi[i] >= RSI_OVERBOUGHT
        rsi_oversold = rsi[i] <= RSI_OVERSOLD
        
        # Determine market bias from 4h EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 4h price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 4h price below EMA50
        
        # Entry conditions
        long_entry = bull_bias and rsi_oversold and volume_confirmed and volatile_enough
        short_entry = bear_bias and rsi_overbought and volume_confirmed and volatile_enough
        
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
            # Exit long when RSI crosses above 50 (mean reversion complete)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when RSI crosses below 50 (mean reversion complete)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals