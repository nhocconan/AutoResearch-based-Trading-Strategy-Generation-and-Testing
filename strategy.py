#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: On daily timeframe, KAMA trend direction + RSI momentum + Choppiness regime filter captures sustained moves while avoiding whipsaws in ranging markets. 
KAMA adapts to volatility, RSI(14) > 55 for long momentum / < 45 for short momentum, and Choppiness Index > 61.8 filters ranging conditions. 
This combination should work in both bull (trend following) and bear (short momentum) markets with low trade frequency to minimize fee drag.
Target: 30-80 trades over 4 years (7-20/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need sufficient data for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === PRIMARY INDICATORS (1d timeframe) ===
    # KAMA: Kaufman Adaptive Moving Average - tracks trend with volatility adaptation
    close_series = pd.Series(close)
    # Efficiency Ratio: |net change| / sum of absolute changes over 10 periods
    change = abs(close_series.diff(10))
    volatility = close_series.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)  # Avoid division by zero
    # Smoothing constants: fastest EMA=2, slowest EMA=30
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # ER-based smoothing constant squared
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close_series.iloc[9]  # Start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc.iloc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]  # Hold previous value if calculation fails
    
    # RSI(14): Relative Strength Index for momentum
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Choppiness Index: measures whether market is choppy (ranging) or trending
    # High CHOP (>61.8) = ranging, Low CHOP (<38.2) = trending
    atr_period = 14
    chop_period = 14
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # ATR
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    # Highest high and lowest low over chop_period
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    # Chop formula: 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(chop_period)
    atr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    range_hl = highest_high - lowest_low
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(chop_period)
    chop_values = chop
    
    # === HTF: 1w trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter (responsive but smooth)
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === SIGNAL GENERATION ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # 25% position size
    bars_since_entry = 0
    
    # Start after warmup period (need 20 for KAMA/RSI/CHOP initialization)
    start_idx = 20
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi_values[i]) or 
            np.isnan(chop_values[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi_values[i]
        chop_val = chop_values[i]
        ema_1w_val = ema_20_1w_aligned[i]
        
        # Long conditions:
        # 1. Price above KAMA (uptrend)
        # 2. RSI > 55 (bullish momentum, not overbought yet)
        # 3. Chop > 61.8 OR Weekly EMA trending up (avoid strong ranging markets unless trend confirms)
        long_condition = (close_val > kama_val) and (rsi_val > 55) and ((chop_val > 61.8) or (close_val > ema_1w_val))
        
        # Short conditions:
        # 1. Price below KAMA (downtrend)
        # 2. RSI < 45 (bearish momentum, not oversold yet)
        # 3. Chop > 61.8 OR Weekly EMA trending down
        short_condition = (close_val < kama_val) and (rsi_val < 45) and ((chop_val > 61.8) or (close_val < ema_1w_val))
        
        # Exit conditions: trend reversal or momentum divergence
        exit_long = (close_val < kama_val) or (rsi_val < 40)  # Price breaks trend or momentum fails
        exit_short = (close_val > kama_val) or (rsi_val > 60)  # Price breaks trend or momentum fails
        
        # Minimum holding period: 3 days to reduce whipsaw and fee drag
        if position != 0 and bars_since_entry < 3:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Entry logic
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        # Exit logic
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0