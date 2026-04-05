#!/usr/bin/env python3
"""
Experiment #8007: 6-hour Camarilla pivot with 1-week trend filter and volume confirmation.
Hypothesis: Price breaking Camarilla R4/S4 levels on 6h with volume >1.8x 20-period MA 
and aligned 1w trend (price above/below 1w EMA50) captures institutional breakouts. 
The 1w timeframe provides long-term trend context to filter counter-trend moves, 
while Camarilla levels from daily data provide precise intraday support/resistance. 
Target: 100-180 total trades over 4 years (25-45/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8007_6h_camarilla1w_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
EMA_WEEKLY_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla levels for given OHLC"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    c = close + (range_val * 1.1 / 12)
    d = close - (range_val * 1.1 / 12)
    l3 = close + (range_val * 1.1 / 6)
    h3 = close - (range_val * 1.1 / 6)
    l4 = close + (range_val * 1.1 / 4)
    h4 = close - (range_val * 1.1 / 4)
    return l3, h3, l4, h4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=EMA_WEEKLY_PERIOD, adjust=False, min_periods=EMA_WEEKLY_PERIOD).mean().values
    price_vs_ema_1w = np.where(close_1w > ema_1w, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_1w_aligned = align_htf_to_ltf(prices, df_1w, price_vs_ema_1w)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        l3, h3, l4, h4 = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        camarilla_l3[i] = l3
        camarilla_h3[i] = h3
        camarilla_l4[i] = l4
        camarilla_h4[i] = h4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, EMA_WEEKLY_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_1w_aligned[i]) or np.isnan(camarilla_l4_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1w EMA
        bull_bias = price_vs_ema_1w_aligned[i] == 1   # 1w close above EMA50
        bear_bias = price_vs_ema_1w_aligned[i] == -1  # 1w close below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions - Camarilla H4/L4 breakouts
        upper_breakout = close[i] > camarilla_h4_aligned[i]
        lower_breakout = close[i] < camarilla_l4_aligned[i]
        
        # Entry conditions
        long_entry = bull_bias and upper_breakout and volume_confirmed
        short_entry = bear_bias and lower_breakout and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals