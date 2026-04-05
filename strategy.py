#!/usr/bin/env python3
"""
Experiment #8174: 1-hour strategy with 4h/1d trend filter and volume confirmation.
Hypothesis: Using 4h for trend direction and 1d for higher-timeframe confirmation, 
combined with volume spikes on 1h for entry timing, reduces whipsaw and captures 
sustained moves in both bull and bear markets. The 1d trend filter adds robustness 
against false breakouts during consolidation, while 4h provides timely trend signals.
Target trade count: 60-150 total over 4 years (15-37/year) to avoid fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8174_1h_4h_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
TREND_PERIOD_4H = 20          # 4h EMA for trend direction
TREND_PERIOD_1D = 50          # 1d EMA for higher-timeframe filter
VOLUME_MA_PERIOD = 20         # Volume moving average
VOLUME_THRESHOLD = 1.5        # Volume spike threshold
SIGNAL_SIZE = 0.20            # Position size (20% of capital)
ATR_PERIOD = 14               # ATR for volatility
ATR_STOP_MULTIPLIER = 2.0     # Stop loss multiplier
SESSION_START_HOUR = 8        # Active session start (UTC)
SESSION_END_HOUR = 20         # Active session end (UTC)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=TREND_PERIOD_4H, adjust=False, min_periods=TREND_PERIOD_4H).mean().values
    trend_4h = np.where(close_4h > ema_4h, 1, -1)  # 1=uptrend, -1=downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d EMA for higher-timeframe filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=TREND_PERIOD_1D, adjust=False, min_periods=TREND_PERIOD_1D).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)  # 1=uptrend, -1=downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Session filter: active hours 8-20 UTC
    hours = prices.index.hour
    in_session = (hours >= SESSION_START_HOUR) & (hours <= SESSION_END_HOUR)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_PERIOD_4H, TREND_PERIOD_1D, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if outside active session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if HTF data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
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
        
        # Determine market bias: require alignment between 4h and 1d trends
        bull_bias = (trend_4h_aligned[i] == 1) and (trend_1d_aligned[i] == 1)
        bear_bias = (trend_4h_aligned[i] == -1) and (trend_1d_aligned[i] == -1)
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_bias and volume_confirmed
        short_entry = bear_bias and volume_confirmed
        
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