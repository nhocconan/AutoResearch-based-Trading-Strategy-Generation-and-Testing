#!/usr/bin/env python3
"""
Experiment #8291: 6-hour Camarilla pivot reversal with 1-day trend filter and volume confirmation.
Hypothesis: Price reversing from Camarilla R4/S3 or S4/R3 levels on 6h with volume >1.5x 20-period MA 
and aligned 1d trend (price above/below 1d EMA50) captures mean reversion in ranging markets and 
breakout failures in trending markets. The 1d trend filter ensures trades align with higher timeframe 
direction, reducing counter-trend losses during strong trends. Targeting 100-200 total trades over 4 years.
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8291_6h_camarilla1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close
    c = close + (range_val * 1.1 / 12)
    d = close - (range_val * 1.1 / 12)
    r3 = close + (range_val * 1.1 / 6)
    s3 = close - (range_val * 1.1 / 6)
    r4 = close + (range_val * 1.1 / 2)
    s4 = close - (range_val * 1.1 / 2)
    return r4, r3, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    price_vs_ema = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Calculate Camarilla levels using previous day's OHLC
    camarilla_r4 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    # Get daily OHLC from 1d data
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla for each day
    for i in range(len(daily_close)):
        r4, r3, s3, s4 = calculate_camarilla(daily_high[i], daily_low[i], daily_close[i])
        camarilla_r4[i] = r4
        camarilla_r3[i] = r3
        camarilla_s3[i] = s3
        camarilla_s4[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(r4_aligned[i]):
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
        
        # Determine market bias from 1d EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d close below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Reversal conditions at Camarilla levels
        # Long: price rejects S3/S4 with bullish bias
        long_reject_s3 = (low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]) and not np.isnan(s3_aligned[i])
        long_reject_s4 = (low[i] <= s4_aligned[i] and close[i] > s4_aligned[i]) and not np.isnan(s4_aligned[i])
        # Short: price rejects R3/R4 with bearish bias
        short_reject_r3 = (high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]) and not np.isnan(r3_aligned[i])
        short_reject_r4 = (high[i] >= r4_aligned[i] and close[i] < r4_aligned[i]) and not np.isnan(r4_aligned[i])
        
        # Entry conditions
        long_entry = bull_bias and (long_reject_s3 or long_reject_s4) and volume_confirmed
        short_entry = bear_bias and (short_reject_r3 or short_reject_r4) and volume_confirmed
        
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