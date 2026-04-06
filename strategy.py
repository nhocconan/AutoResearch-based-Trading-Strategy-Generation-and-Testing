#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour RSI(14) with Bollinger Band(20,2) and volume confirmation.
# RSI < 30 + price near lower BB + volume spike = long signal.
# RSI > 70 + price near upper BB + volume spike = short signal.
# Uses 12h EMA(50) as trend filter: only take longs when above 12h EMA, shorts when below.
# Designed for mean reversion in ranging markets with trend filter to avoid counter-trend trades.
# Target: 80-150 trades over 4 years (20-38/year).
name = "exp_14159_6h_rsi14_bb20_12ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data for EMA(50) trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h close
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14)
    rsi = calculate_rsi(close, 14)
    
    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for BB/volume, 14 for RSI/ATR, 50 for EMA)
    start = max(20, 14, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(rsi[i]) or np.isnan(sma_20[i]) or np.isnan(std_20[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Mean reversion signals with volume and EMA filter
        # Long: RSI < 30 + price near lower BB + volume spike + above 12h EMA
        # Short: RSI > 70 + price near upper BB + volume spike + below 12h EMA
        near_lower_bb = close[i] <= lower_bb[i] + (0.1 * (sma_20[i] - lower_bb[i]))
        near_upper_bb = close[i] >= upper_bb[i] - (0.1 * (upper_bb[i] - sma_20[i]))
        
        long_signal = (rsi[i] < 30) and near_lower_bb and vol_filter[i] and (close[i] > ema_50_aligned[i])
        short_signal = (rsi[i] > 70) and near_upper_bb and vol_filter[i] and (close[i] < ema_50_aligned[i])
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or RSI > 50 (mean reversion complete)
            if close[i] <= stop_price or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or RSI < 50 (mean reversion complete)
            if close[i] >= stop_price or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals