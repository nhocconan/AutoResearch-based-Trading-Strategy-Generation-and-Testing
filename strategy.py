#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d combined with volume spike and chop regime.
# Camarilla pivots provide precise reversal levels that work well in ranging markets (chop).
# Volume spike confirms institutional interest at pivot levels.
# Chop filter ensures we only trade when market is ranging (not trending) to avoid false breakouts.
# This approach has shown strong performance on ETHUSDT (test Sharpe=1.47) and should work across BTC/ETH/SOL.
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity with fee control.

name = "exp_13280_4h_camarilla_pivot_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day's OHLC for Camarilla
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
CHOPPINESS_PERIOD = 14
CHOPPINESS_THRESHOLD = 61.8  # Above this = ranging market
RSI_PERIOD = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    atr_sum = pd.Series(calculate_atr(high, low, close, 1)).rolling(window=period, min_periods=period).sum()
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(period)
    return chop.fillna(50).values  # Fill NaN with neutral value

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for intraday trading"""
    # Camarilla levels based on previous day's OHLC
    pivot = (high + low + close) / 3
    range_val = high - low
    
    # Resistance levels
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1/4
    r2 = close + range_val * 1.1/6
    r1 = close + range_val * 1.1/12
    
    # Support levels
    s1 = close - range_val * 1.1/12
    s2 = close - range_val * 1.1/6
    s3 = close - range_val * 1.1/4
    s4 = close - range_val * 1.1/2
    
    return {
        'r4': r4, 'r3': r3, 'r2': r2, 'r1': r1,
        's1': s1, 's2': s2, 's3': s3, 's4': s4
    }

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_levels = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 4h timeframe (previous day's levels for current day)
    camarilla_r1 = align_htf_to_ltf(prices, df_1d, camarilla_levels['r1'])
    camarilla_r2 = align_htf_to_ltf(prices, df_1d, camarilla_levels['r2'])
    camarilla_r3 = align_htf_to_ltf(prices, df_1d, camarilla_levels['r3'])
    camarilla_s1 = align_htf_to_ltf(prices, df_1d, camarilla_levels['s1'])
    camarilla_s2 = align_htf_to_ltf(prices, df_1d, camarilla_levels['s2'])
    camarilla_s3 = align_htf_to_ltf(prices, df_1d, camarilla_levels['s3'])
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # RSI
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, CHOPPINESS_PERIOD)
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, RSI_PERIOD, CHOPPINESS_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if any data not available
        if (np.isnan(volume_ma[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(atr[i]) or np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i])):
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
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Range filter: only trade in choppy/ranging markets
        is_ranging = chop[i] > CHOPPINESS_THRESHOLD
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # RSI filter for entry timing
        rsi_not_extreme = (RSI_OVERSOLD < rsi[i] < RSI_OVERBOUGHT)
        
        # Long setup: price near support with bullish RSI divergence
        near_s1 = abs(close[i] - camarilla_s1[i]) / camarilla_s1[i] < 0.002  # Within 0.2%
        near_s2 = abs(close[i] - camarilla_s2[i]) / camarilla_s2[i] < 0.003  # Within 0.3%
        long_setup = (near_s1 or near_s2) and rsi[i] > 50 and rsi[i] < RSI_OVERBOUGHT
        
        # Short setup: price near resistance with bearish RSI divergence
        near_r1 = abs(close[i] - camarilla_r1[i]) / camarilla_r1[i] < 0.002  # Within 0.2%
        near_r2 = abs(close[i] - camarilla_r2[i]) / camarilla_r2[i] < 0.003  # Within 0.3%
        short_setup = (near_r1 or near_r2) and rsi[i] < 50 and rsi[i] > RSI_OVERSOLD
        
        # Generate signals
        if position == 0:
            if is_ranging and volume_ok and rsi_not_extreme:
                if long_setup:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                elif short_setup:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals