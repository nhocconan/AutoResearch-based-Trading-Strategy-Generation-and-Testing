#!/usr/bin/env python3
"""
Experiment #9254: 1h timeframe with 4h/1d multi-timeframe trend filter
Hypothesis: Use 4h for trend direction, 1d for regime filter, 1h for precise entry timing.
Target 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
Works in bull (trend following) and bear (mean reversion in range).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_9254_1h_4h_trend_1d_regime_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_FAST = 9
EMA_SLOW = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    ema_fast_4h = pd.Series(close_4h).ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    ema_slow_4h = pd.Series(close_4h).ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    trend_4h = np.where(ema_fast_4h > ema_slow_4h, 1, -1)  # 1=uptrend, -1=downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d RSI for regime filter
    close_1d = df_1d['close'].values
    rsi_1d = calculate_rsi(close_1d, RSI_PERIOD)
    # Regime: 1=trending (RSI extreme), 0=ranging (RSI middle)
    regime_1d = np.where((rsi_1d > RSI_OVERBOUGHT) | (rsi_1d < RSI_OVERSOLD), 1, 0)
    regime_1d_aligned = align_htf_to_ltf(prices, df_1d, regime_1d)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA for entry timing
    ema_fast = pd.Series(close).ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    ema_slow = pd.Series(close).ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    
    # RSI for mean reversion signals
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_FAST, EMA_SLOW, RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(regime_1d_aligned[i]):
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
        
        # Determine market conditions
        uptrend = trend_4h_aligned[i] == 1
        downtrend = trend_4h_aligned[i] == -1
        trending_regime = regime_1d_aligned[i] == 1
        ranging_regime = regime_1d_aligned[i] == 0
        
        # EMA crossover signals
        ema_bullish = ema_fast[i] > ema_slow[i]
        ema_bearish = ema_fast[i] < ema_slow[i]
        
        # RSI mean reversion signals
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry logic: trend following in trending regime, mean reversion in ranging regime
        long_entry = False
        short_entry = False
        
        if trending_regime:
            # Trend following: follow 4h trend
            if uptrend and ema_bullish and volume_confirmed:
                long_entry = True
            if downtrend and ema_bearish and volume_confirmed:
                short_entry = True
        else:  # ranging_regime
            # Mean reversion: fade extremes
            if rsi_oversold and ema_bullish and volume_confirmed:
                long_entry = True
            if rsi_overbought and ema_bearish and volume_confirmed:
                short_entry = True
        
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