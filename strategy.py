#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12819_6d_12h_1d_adx_cci_trend"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
CCI_PERIOD = 20
CCI_OVERBOUGHT = 100
CCI_OVERSOLD = -100
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.25
MAX_HOLD_BARS = 48  # Max 12 days (48 * 6h)

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate ADX indicator"""
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Wilder's smoothing
    atr = np.concatenate([[np.nan], pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values])
    plus_di = 100 * pd.Series(np.concatenate([[np.nan], plus_dm])).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(np.concatenate([[np.nan], minus_dm])).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def calculate_cci(high, low, close, period):
    """Calculate Commodity Channel Index"""
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp).rolling(window=period, min_periods=period).mean()
    mad = pd.Series(tp).rolling(window=period, min_periods=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for ADX and CCI
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily ADX and CCI
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    adx_daily = calculate_adx(high_d, low_d, close_d, ADX_PERIOD)
    cci_daily = calculate_cci(high_d, low_d, close_d, CCI_PERIOD)
    
    # Align to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx_daily)
    cci_aligned = align_htf_to_ltf(prices, df_daily, cci_daily)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(ADX_PERIOD, CCI_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if ADX or CCI not available
        if np.isnan(adx_aligned[i]) or np.isnan(cci_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Time-based exit to prevent overtrading
        if bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
        
        # Trend condition: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # CCI conditions for entry
        cci_buy = cci_aligned[i] < CCI_OVERSOLD  # Oversold
        cci_sell = cci_aligned[i] > CCI_OVERBOUGHT  # Overbought
        
        # Generate signals
        if position == 0:
            if trending and cci_buy:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            elif trending and cci_sell:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals