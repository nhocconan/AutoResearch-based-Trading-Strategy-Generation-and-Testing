#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s strategy combining weekly trend from 1w EMA200 with intraday mean reversion using RSI(2) on 6h timeframe.
# Long when: price above weekly EMA200 (bullish regime) AND RSI(2) < 10 (oversold) AND volume above average.
# Short when: price below weekly EMA200 (bearish regime) AND RSI(2) > 90 (overbought) AND volume above average.
# Uses ATR-based stop loss to manage risk.
# Weekly EMA200 provides regime filter to avoid counter-trend trades in strong trends.
# RSI(2) captures short-term mean reversion within the regime.
# Volume confirms conviction behind the move.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13875_6h_weekly_ema200_rsi2_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
RSI_PERIOD = 2
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
EMA_PERIOD_WEEKLY = 200
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(close, period):
    """Calculate RSI with proper Wilder smoothing"""
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
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data for weekly EMA200 ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=EMA_PERIOD_WEEKLY, adjust=False, min_periods=EMA_PERIOD_WEEKLY).mean().values
    
    # Align weekly EMA200 to 6h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # 6h data for RSI, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(2) for mean reversion signals
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Mean reversion signals from RSI(2)
        rsi_oversold = rsi[i] < 10
        rsi_overbought = rsi[i] > 90
        
        # Regime filter from weekly EMA200
        price_above_weekly_ema = close[i] > ema_200_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_200_1w_aligned[i]
        
        # Generate signals
        if position == 0:
            # Long: bullish regime + oversold + volume
            if price_above_weekly_ema and rsi_oversold and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: bearish regime + overbought + volume
            elif price_below_weekly_ema and rsi_overbought and volume_ok:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on RSI > 50 (mean reversion complete) or stop loss
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on RSI < 50 (mean reversion complete) or stop loss
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals