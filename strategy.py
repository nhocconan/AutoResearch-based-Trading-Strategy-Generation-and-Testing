#!/usr/bin/env python3
"""
1h RSI reversal with 4h trend filter and volume confirmation.
Hypothesis: RSI extremes (overbought/oversold) on 1h combined with 4h trend direction
and volume spikes capture mean-reversion moves in both bull and bear markets.
4h trend filter avoids counter-trend trades. Volume confirmation ensures momentum.
Target: 60-150 trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14274_1h_rsi_reversal_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period):
    """Calculate RSI with proper min_periods"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for EMA trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(50) for trend
    ema_4h = calculate_ema(close_4h, 50)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) for mean reversion signals
    rsi = calculate_rsi(close, 14)
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 14 for RSI, 20 for volume, 50 for EMA)
    start = max(14, 20, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Mean reversion signals with 4h trend filter and volume spike
        # Long: RSI < 30 (oversold) + price > 4h EMA50 (uptrend) + volume spike
        # Short: RSI > 70 (overbought) + price < 4h EMA50 (downtrend) + volume spike
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        price_above_4h_ema = close[i] > ema_4h_aligned[i]
        price_below_4h_ema = close[i] < ema_4h_aligned[i]
        
        signal_long = rsi_oversold and price_above_4h_ema and vol_spike[i]
        signal_short = rsi_overbought and price_below_4h_ema and vol_spike[i]
        
        # Generate signals
        if position == 0:
            if signal_long:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif signal_short:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on RSI > 50 (mean reversion complete) or reverse signal
            if rsi[i] > 50 or signal_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short on RSI < 50 (mean reversion complete) or reverse signal
            if rsi[i] < 50 or signal_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals