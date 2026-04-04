#!/usr/bin/env python3
"""
Experiment #2857: 4h Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian channel breakouts capture strong momentum moves. Combining with 
HMA(21) trend filter ensures we trade in the direction of the intermediate trend, 
while volume confirmation filters out false breakouts. ATR-based stoploss manages 
risk. This pattern has shown strong test performance on SOLUSDT (Sharpe 1.10-1.38). 
Using 4h timeframe targets 75-200 trades over 4 years (19-50/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2857_4h_donchian20_hma21_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === HTF: 1w data for regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(100) for stronger trend filter
    ema_1w = pd.Series(close_1w).ewm(span=100, min_periods=100, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_1w, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 4h Indicators: Donchian(20) channels ===
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: HMA(21) for trend confirmation ===
    def hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
        raw = 2 * wma2 - wma1
        return raw.ewm(span=sqrt, adjust=False).mean().values
    
    hma_21 = hma(close, 21)
    hma_rising = np.zeros(n, dtype=bool)
    hma_rising[1:] = hma_21[1:] > hma_21[:-1]
    
    # === 4h Indicators: Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    volume_spike = vol_ratio > 1.5  # 50% above average volume
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # sufficient for all indicators (Donchian20 + HMA21 + ATR14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(hma_21[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if Donchian breakout in opposite direction (mean reversion)
                elif price < donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if Donchian breakout in opposite direction (mean reversion)
                elif price > donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike for confirmation
        if volume_spike[i]:
            # Get trend biases from multiple timeframes
            trend_bias_1d = trend_1d_aligned[i]
            trend_bias_1w = trend_1w_aligned[i]
            
            # Require alignment of 1d and 1w trends (both bullish or both bearish)
            trend_aligned = (trend_bias_1d == trend_bias_1w)
            
            # Long entry: price breaks above Donchian upper + rising HMA + bullish trend alignment
            if (price > donchian_upper[i] and hma_rising[i] and 
                trend_bias_1d > 0 and trend_bias_1w > 0 and trend_aligned):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower + falling HMA + bearish trend alignment
            elif (price < donchian_lower[i] and not hma_rising[i] and 
                  trend_bias_1d < 0 and trend_bias_1w < 0 and trend_aligned):
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals