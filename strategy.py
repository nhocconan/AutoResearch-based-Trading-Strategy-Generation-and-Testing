#!/usr/bin/env python3
# 4h_Exponential_Rebound_v1
# Hypothesis: Uses price deviation from 4h EMA20 as a mean-reversion signal in trending markets.
# When price deviates significantly below EMA20 with bullish momentum (MACD histogram rising),
# go long; when price deviates significantly above EMA20 with bearish momentum,
# go short. Uses 12h EMA50 as trend filter to avoid counter-trend trades.
# Includes volatility filter (ATR-based) to avoid low-volatility chop.
# Designed for low trade frequency by requiring multiple confluence factors.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

name = "4h_Exponential_Rebound_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4h EMA20 (dynamic mean) ---
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # --- Price deviation from EMA20 (normalized by ATR-like measure) ---
    # Use 10-period range as volatility normalizer
    high_low = pd.Series(high - low)
    vol_norm = high_low.rolling(window=10, min_periods=10).mean().values
    vol_norm = np.where(vol_norm == 0, 1e-8, vol_norm)  # avoid division by zero
    price_dev = (close - ema20) / vol_norm  # positive = above EMA20
    
    # --- MACD histogram (12,26,9) for momentum ---
    ema12 = close_series.ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = close_series.ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    # Rising histogram = bullish momentum, falling = bearish
    macd_hist_rising = np.diff(macd_hist, prepend=macd_hist[0]) > 0
    macd_hist_falling = np.diff(macd_hist, prepend=macd_hist[0]) < 0
    
    # --- 12h EMA50 trend filter ---
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    # Uptrend: price above EMA50, Downtrend: price below EMA50
    uptrend = close > ema50_12h_aligned
    downtrend = close < ema50_12h_aligned
    
    # --- Volatility filter: avoid low volatility chop ---
    # Use 20-period ATR normalized by price
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    vol_filter = (atr / close) > 0.01  # only trade when volatility > 1% of price
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(price_dev[i]) or np.isnan(macd_hist_rising[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_filter[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price significantly below EMA20, bullish momentum, uptrend, sufficient volatility
            if (price_dev[i] < -0.8 and   # price well below EMA20
                macd_hist_rising[i] and   # bullish momentum building
                uptrend[i] and            # in uptrend (12h EMA50)
                vol_filter[i]):           # sufficient volatility
                signals[i] = 0.25
                position = 1
            # Short: price significantly above EMA20, bearish momentum, downtrend, sufficient volatility
            elif (price_dev[i] > 0.8 and    # price well above EMA20
                  macd_hist_falling[i] and  # bearish momentum building
                  downtrend[i] and          # in downtrend (12h EMA50)
                  vol_filter[i]):           # sufficient volatility
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: mean reversion or trend change
            if position == 1:
                # Exit long: price returns to EMA20 or trend turns down
                if price_dev[i] > -0.2 or not uptrend[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to EMA20 or trend turns up
                if price_dev[i] < 0.2 or not downtrend[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals