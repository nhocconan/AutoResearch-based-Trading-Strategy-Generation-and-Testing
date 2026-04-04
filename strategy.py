#!/usr/bin/env python3
"""
Experiment #3494: 1h Donchian Breakout + 4h/1d Trend Filter + Volume Confirmation + Session Filter
HYPOTHESIS: 1h Donchian(20) breakouts aligned with 4h/1d trend direction and volume confirmation capture medium-term momentum while avoiding counter-trend whipsaws. Session filter (08-20 UTC) reduces noise. Position size 0.20. Target: 75-150 total trades over 4 years (19-37/year).
Uses 4h/1d for trend direction and bias, 1h only for entry timing and execution. Works in bull (continuation from uptrend) and bear (continuation from downtrend) via trend filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3494_1h_donchian20_4h1d_trend_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC) for filtering
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) for trend direction
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 4h price above/below EMA50 = bullish/bearish bias
    trend_bias_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_bias_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_bias_4h)
    
    # === HTF: 1d data for higher timeframe trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(100) for higher timeframe trend
    ema_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d price above/below EMA100 = stronger bias confirmation
    trend_bias_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_bias_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # === 1h Indicators: Donchian channels (20-period) for entry timing ===
    lookback_1h = 20
    highest_high_1h = pd.Series(high).rolling(window=lookback_1h, min_periods=lookback_1h).max().values
    lowest_low_1h = pd.Series(low).rolling(window=lookback_1h, min_periods=lookback_1h).min().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(100, lookback_1h, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC only ---
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
            
        # --- Data Validity Check ---
        if (np.isnan(highest_high_1h[i]) or np.isnan(lowest_low_1h[i]) or
            np.isnan(trend_bias_4h_aligned[i]) or np.isnan(trend_bias_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Require both 4h and 1d trend to align for stronger bias
            bullish_bias = (trend_bias_4h_aligned[i] > 0 and trend_bias_1d_aligned[i] > 0)
            bearish_bias = (trend_bias_4h_aligned[i] < 0 and trend_bias_1d_aligned[i] < 0)
            
            # Long entry: price breaks above 1h Donchian high with bullish bias
            if (price > highest_high_1h[i] and bullish_bias):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 1h Donchian low with bearish bias
            elif (price < lowest_low_1h[i] and bearish_bias):
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