#!/usr/bin/env python3
"""
Experiment #8314: 1-hour mean reversion with 4h/1d trend filter and volume exhaustion.
Hypothesis: In ranging markets (common in 2025), price reverts to VWAP after extreme moves 
identified by RSI divergence and volume exhaustion. The 4h trend filter ensures we only 
trade mean reversion in the direction of higher timeframe momentum, while 1d trend 
avoids trading against major trend. Session filter (08-20 UTC) reduces noise. Targets 
15-37 trades/year to minimize fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8314_1h_meanrev_4h_1d_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VWAP_DIST_THRESHOLD = 0.015  # 1.5% from VWAP
VOLUME_EXHAUSTION = 0.6  # volume < 60% of 20-period average
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h trend: price above/below EMA50
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h = np.where(close_4h > ema_4h, 1, -1)  # 1=uptrend, -1=downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)  # 1=uptrend, -1=downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # VWAP approximation (session-based)
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / (volume.cumsum() + 1e-10)
    vwap_dist = (close - vwap) / vwap  # percentage distance from VWAP
    
    # Volume exhaustion: current volume vs 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (volume_ma + 1e-10)
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(RSI_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Check session
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Check stoploss
        if position == 1 and close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Mean reversion conditions
        rsi_extreme = rsi[i] >= RSI_OVERBOUGHT or rsi[i] <= RSI_OVERSOLD
        vwap_extreme = abs(vwap_dist[i]) >= VWAP_DIST_THRESHOLD
        volume_exhausted = volume_ratio[i] <= VOLUME_EXHAUSTION
        
        # Determine mean reversion direction
        if rsi[i] >= RSI_OVERBOUGHT and vwap_dist[i] > 0:
            # Overbought and above VWAP -> expect reversion down
            mean_reversion_signal = -1
        elif rsi[i] <= RSI_OVERSOLD and vwap_dist[i] < 0:
            # Oversold and below VWAP -> expect reversion up
            mean_reversion_signal = 1
        else:
            mean_reversion_signal = 0
        
        # Only take signals that align with 4h trend (trade with higher timeframe momentum)
        if mean_reversion_signal != 0:
            # For long signals, we want 4h uptrend or at least not strong downtrend
            # For short signals, we want 4h downtrend or at least not strong uptrend
            if mean_reversion_signal == 1 and trend_4h_aligned[i] >= -1:  # allow long in uptrend or sideways
                signal_aligned = 1
            elif mean_reversion_signal == -1 and trend_4h_aligned[i] <= 1:  # allow short in downtrend or sideways
                signal_aligned = -1
            else:
                signal_aligned = 0
        else:
            signal_aligned = 0
        
        # Additional 1d trend filter: avoid trading strongly against daily trend
        if signal_aligned == 1 and trend_1d_aligned[i] == -1:
            # Strong daily downtrend, avoid longs
            signal_aligned = 0
        elif signal_aligned == -1 and trend_1d_aligned[i] == 1:
            # Strong daily uptrend, avoid shorts
            signal_aligned = 0
        
        # Generate signals
        if position == 0:
            if signal_aligned != 0 and in_session:
                signals[i] = signal_aligned * SIGNAL_SIZE
                position = signal_aligned
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals