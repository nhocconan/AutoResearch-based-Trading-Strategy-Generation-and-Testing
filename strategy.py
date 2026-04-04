#!/usr/bin/env python3
"""
Experiment #3254: 1h Donchian Breakout + 4h/1d Trend + Volume Spike + Session Filter
HYPOTHESIS: 1h Donchian(20) breakouts with volume confirmation and 4h/1d trend filters capture medium-term swings.
Primary timeframe 1h for precise entry timing, HTF (4h/1d) for signal direction to reduce noise and overtrading.
Session filter (08-20 UTC) avoids low-liquidity periods. Discrete position sizing (0.20) minimizes fee churn.
Designed for both bull (breakout continuation) and bear (mean reversion from extremes) via price channels and volatility filters.
Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3254_1h_donchian20_4h_1d_trend_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === HTF: 4h data for Donchian trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels (20-period) on 4h
    lookback_4h = 20
    highest_high_4h = pd.Series(high_4h).rolling(window=lookback_4h, min_periods=lookback_4h).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=lookback_4h, min_periods=lookback_4h).min().values
    htf_trend_up = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    htf_trend_down = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # === HTF: 1d data for EMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: Donchian channels (20-period) for breakout ===
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
    SIZE = 0.20  # 20% position size (discrete level to minimize fee churn)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback_1h, 20, 14, 50)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high_1h[i]) or np.isnan(lowest_low_1h[i]) or
            np.isnan(htf_trend_up[i]) or np.isnan(htf_trend_down[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (stoploss)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters 1h Donchian channel (mean reversion)
                elif price <= highest_high_1h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (stoploss)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters 1h Donchian channel (mean reversion)
                elif price >= lowest_low_1h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) for confirmation
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # HTF trend filters: 
            # 4h: price above 4h Donchian high = bullish, below 4h Donchian low = bearish
            # 1d: price above 1d EMA50 = bullish, below = bearish
            price_vs_4h_high = price - htf_trend_up[i]
            price_vs_4h_low = price - htf_trend_down[i]
            price_vs_ema_1d = price - ema_1d_aligned[i]
            
            # Long entry: price breaks above 1h Donchian high with bullish HTF alignment
            if (price > highest_high_1h[i] and 
                price_vs_4h_high > 0 and  # above 4h Donchian high (bullish 4h trend)
                price_vs_ema_1d > 0):     # above 1d EMA50 (bullish 1d trend)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 1h Donchian low with bearish HTF alignment
            elif (price < lowest_low_1h[i] and 
                  price_vs_4h_low < 0 and   # below 4h Donchian low (bearish 4h trend)
                  price_vs_ema_1d < 0):     # below 1d EMA50 (bearish 1d trend)
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