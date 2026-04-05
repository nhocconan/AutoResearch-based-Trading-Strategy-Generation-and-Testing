#!/usr/bin/env python3
"""
Experiment #8899: 6h Camarilla pivot + volume + trend filter.
Hypothesis: Camarilla levels (R3/S3 for reversal, R4/S4 for breakout) provide high-probability reversal/breakout zones. Volume confirms participation, and 12h EMA filter ensures directional alignment with higher timeframe trend. Works in both bull and bear markets by adapting to reversal or breakout logic based on volatility regime.
Targets 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8899_6h_camarilla_pivot_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_LOOKBACK = 10  # Lookback period for prior day OHLC
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
TREND_PERIOD = 30
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for given high, low, close
    Returns: (R4, R3, S3, S4)
    """
    range_val = high - low
    if range_val <= 0:
        return np.nan, np.nan, np.nan, np.nan
    close_val = close
    R4 = close_val + (range_val * 1.1 / 2)
    R3 = close_val + (range_val * 1.1 / 4)
    S3 = close_val - (range_val * 1.1 / 4)
    S4 = close_val - (range_val * 1.1 / 2)
    return R4, R3, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=TREND_PERIOD, adjust=False, min_periods=TREND_PERIOD).mean().values
    
    # Price relative to 12h EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_12h > ema_12h, 1, 
                     np.where(close_12h < ema_12h, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_12h, price_vs_ema)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate prior period OHLC for Camarilla (using previous day's data)
    # For 6h chart, we use the prior day's OHLC (4 periods back = 24h)
    lookback = CAMARILLA_LOOKBACK  # 4 periods = 24h for 6h chart
    prev_high = np.roll(high, lookback)
    prev_low = np.roll(low, lookback)
    prev_close = np.roll(close, lookback)
    
    # Calculate Camarilla levels for each bar
    camarilla_R4 = np.full(n, np.nan)
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    camarilla_S4 = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if not np.isnan(prev_high[i]) and not np.isnan(prev_low[i]) and not np.isnan(prev_close[i]):
            R4, R3, S3, S4 = calculate_camarilla(prev_high[i], prev_low[i], prev_close[i])
            camarilla_R4[i] = R4
            camarilla_R3[i] = R3
            camarilla_S3[i] = S3
            camarilla_S4[i] = S4
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(CAMARILLA_LOOKBACK, TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Skip if Camarilla levels not available
        if np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]):
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
        
        # Determine market bias from 12h EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 12h price above EMA30
        bear_bias = price_vs_ema_aligned[i] == -1  # 12h price below EMA30
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Price action at Camarilla levels
        price = close[i]
        
        # Reversal logic at R3/S3 (fade extreme moves)
        at_R3 = abs(price - camarilla_R3[i]) < (0.1 * camarilla_R3[i])  # Within 0.1% of R3
        at_S3 = abs(price - camarilla_S3[i]) < (0.1 * camarilla_S3[i])  # Within 0.1% of S3
        
        # Breakout logic at R4/S4 (continue strong moves)
        above_R4 = price > camarilla_R4[i]
        below_S4 = price < camarilla_S4[i]
        
        # Entry conditions
        # Long: reversal at S3 in bullish trend OR breakout above R4 with volume
        long_reversal = bull_bias and at_S3 and volume_confirmed
        long_breakout = bull_bias and above_R4 and volume_confirmed
        
        # Short: reversal at R3 in bearish trend OR breakdown below S4 with volume
        short_reversal = bear_bias and at_R3 and volume_confirmed
        short_breakout = bear_bias and below_S4 and volume_confirmed
        
        long_entry = long_reversal or long_breakout
        short_entry = short_reversal or short_breakout
        
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