#!/usr/bin/env python3
"""
Experiment #3476: 12h Donchian(20) Breakout + 1d Trend Filter + Volume Spike + ATR Stoploss
HYPOTHESIS: 12h Donchian breakouts with volume confirmation and 1d EMA(50) trend alignment capture medium-term momentum with controlled trade frequency.
Works in bull markets via trend continuation and bear markets via mean reversion from extremes (Donchian channel acts as dynamic support/resistance).
Uses 1d for signal direction, 12h only for entry timing. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3476_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter and Donchian (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channels (20-period) on 1d for trend filter
    lookback_1d = 20
    highest_high_1d = pd.Series(high_1d).rolling(window=lookback_1d, min_periods=lookback_1d).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=lookback_1d, min_periods=lookback_1d).min().values
    highest_high_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_high_1d)
    lowest_low_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_1d)
    
    # === 12h Indicators: Donchian channels (20-period) for entry timing ===
    lookback_12h = 20
    highest_high_12h = pd.Series(high).rolling(window=lookback_12h, min_periods=lookback_12h).max().values
    lowest_low_12h = pd.Series(low).rolling(window=lookback_12h, min_periods=lookback_12h).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level to minimize fee churn)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback_12h, lookback_1d, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_12h[i]) or np.isnan(lowest_low_12h[i]) or
            np.isnan(highest_high_1d_aligned[i]) or np.isnan(lowest_low_1d_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
                # Exit if price re-enters 12h Donchian channel (mean reversion)
                elif price <= highest_high_12h[i]:
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
                # Exit if price re-enters 12h Donchian channel (mean reversion)
                elif price >= lowest_low_12h[i]:
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
            # 1d Donchian trend filter: only long above 1d highest, short below 1d lowest
            price_vs_1d_high = price - highest_high_1d_aligned[i]
            price_vs_1d_low = price - lowest_low_1d_aligned[i]
            
            # 1d EMA trend filter: only long above EMA, short below EMA
            price_vs_ema = price - ema_1d_aligned[i]
            
            # Long entry: price breaks above 12h Donchian high with bullish 1d trend
            if (price > highest_high_12h[i] and 
                price_vs_1d_high > 0 and 
                price_vs_ema > 0):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 12h Donchian low with bearish 1d trend
            elif (price < lowest_low_12h[i] and 
                  price_vs_1d_low < 0 and 
                  price_vs_ema < 0):
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