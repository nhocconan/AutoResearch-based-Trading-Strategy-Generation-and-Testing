#!/usr/bin/env python3
"""
Experiment #3488: 12h Donchian Breakout + 1w Trend Filter + Volume Spike + ATR Stop
HYPOTHESIS: 12h Donchian(20) breakouts with volume confirmation and 1w trend alignment capture medium-term momentum while avoiding overtrading. Uses 1w for signal direction, 12h only for entry timing. Works in bull (trend continuation) and bear (mean reversion from extremes) via price channels. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3488_12h_donchian20_1w_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 1w
    lookback_1w = 20
    highest_high_1w = pd.Series(high_1w).rolling(window=lookback_1w, min_periods=lookback_1w).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=lookback_1w, min_periods=lookback_1w).min().values
    highest_high_1w_aligned = align_htf_to_ltf(prices, df_1w, highest_high_1w)
    lowest_low_1w_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_1w)
    
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
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback_12h, lookback_1w, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_12h[i]) or np.isnan(lowest_low_12h[i]) or
            np.isnan(highest_high_1w_aligned[i]) or np.isnan(lowest_low_1w_aligned[i]) or
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
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # 1w Donchian trend filter: only long above 1w highest, short below 1w lowest
            price_vs_1w_high = price - highest_high_1w_aligned[i]
            price_vs_1w_low = price - lowest_low_1w_aligned[i]
            
            # Long entry: price breaks above 12h Donchian high with bullish 1w trend
            if (price > highest_high_12h[i] and 
                price_vs_1w_high > 0):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 12h Donchian low with bearish 1w trend
            elif (price < lowest_low_12h[i] and 
                  price_vs_1w_low < 0):
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

</think>