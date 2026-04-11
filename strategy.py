#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1w Supertrend + 1d volume confirmation + price above/below 1d EMA50
# - Supertrend on 1w timeframe (ATR period=10, multiplier=3) to determine long-term trend
# - Long when: price > Supertrend (uptrend) AND price > 1d EMA50 AND volume > 1.5x 20-period average
# - Short when: price < Supertrend (downtrend) AND price < 1d EMA50 AND volume > 1.5x 20-period average
# - Uses 12h timeframe for entries, targeting 12-37 trades/year (50-150 total over 4 years)
# - Supertrend provides strong trend filter, reducing false signals in choppy markets
# - Volume confirmation ensures momentum behind moves
# - 1d EMA50 avoids counter-trend trades in strong trends
# - Discrete position sizing (±0.25) to limit drawdown and reduce fee churn

name = "12h_1w_supertrend_volume_ema50_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for Supertrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Calculate Supertrend on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR
    atr_period = 10
    atr = np.full_like(tr, np.nan, dtype=float)
    for i in range(len(tr)):
        if i < atr_period:
            atr[i] = np.nan
        elif i == atr_period:
            atr[i] = np.nanmean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Supertrend calculation
    factor = 3
    hl2 = (high_1w + low_1w) / 2
    upperband = hl2 + (factor * atr)
    lowerband = hl2 - (factor * atr)
    
    supertrend = np.full_like(close_1w, np.nan, dtype=float)
    uptrend = np.full_like(close_1w, True, dtype=bool)
    
    for i in range(1, len(close_1w)):
        if np.isnan(atr[i]) or np.isnan(atr[i-1]):
            supertrend[i] = np.nan
            uptrend[i] = uptrend[i-1]
            continue
            
        if close_1w[i] > upperband[i-1]:
            uptrend[i] = True
        elif close_1w[i] < lowerband[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if not uptrend[i] and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        supertrend[i] = lowerband[i] if uptrend[i] else upperband[i]
    
    # Align Supertrend and uptrend to 12h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    uptrend_aligned = align_htf_to_ltf(prices, df_1w, uptrend.astype(float)) > 0.5
    
    # Load 1d data ONCE before loop for EMA50 and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute 12h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(supertrend_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Trend filters
        price_above_supertrend = price_close > supertrend_aligned[i]  # Uptrend
        price_below_supertrend = price_close < supertrend_aligned[i]  # Downtrend
        price_above_ema50 = price_close > ema50_1d_aligned[i]
        price_below_ema50 = price_close < ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Uptrend (price > Supertrend) + price above 1d EMA50 + volume confirmation
        if price_above_supertrend and price_above_ema50 and vol_confirm:
            enter_long = True
        
        # Short: Downtrend (price < Supertrend) + price below 1d EMA50 + volume confirmation
        if price_below_supertrend and price_below_ema50 and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite trend or price crosses 1d EMA50
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if downtrend OR price crosses below 1d EMA50
            exit_long = price_below_supertrend or (not price_above_ema50)
        elif position == -1:
            # Exit short if uptrend OR price crosses above 1d EMA50
            exit_short = price_above_supertrend or (not price_below_ema50)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals