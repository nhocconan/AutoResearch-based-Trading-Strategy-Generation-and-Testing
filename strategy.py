#!/usr/bin/env python3
"""
Experiment #3584: 1d Donchian Breakout + 1w Weekly Pivot + Volume Confirmation
HYPOTHESIS: Daily Donchian(20) breakouts aligned with weekly pivot direction and volume spikes capture institutional momentum in both bull and bear markets. Weekly pivot provides key support/resistance from smart money, while volume confirms breakout authenticity. Uses discrete position sizing (0.25) to minimize fee churn and ATR-based trailing stops for risk control. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3584_1d_donchian20_1w_pivot_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1w data for weekly pivot (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    lookback_week = 1  # For weekly data, lookback of 1 means prior completed week
    prior_week_high = pd.Series(high_1w).rolling(window=lookback_week, min_periods=lookback_week).max().shift(1).values
    prior_week_low = pd.Series(low_1w).rolling(window=lookback_week, min_periods=lookback_week).min().shift(1).values
    prior_week_close = pd.Series(close_1w).rolling(window=lookback_week, min_periods=lookback_week).mean().shift(1).values
    
    # Weekly pivot formula: P = (H + L + C) / 3
    weekly_pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
    # Resistance 1: R1 = 2*P - L
    r1 = 2 * weekly_pivot - prior_week_low
    # Support 1: S1 = 2*P - H
    s1 = 2 * weekly_pivot - prior_week_high
    
    # Align all pivot levels to 1d timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === 1d Indicators: Donchian channels (20-period) for entry timing ===
    lookback_1d = 20
    highest_high_1d = pd.Series(high).rolling(window=lookback_1d, min_periods=lookback_1d).max().values
    lowest_low_1d = pd.Series(low).rolling(window=lookback_1d, min_periods=lookback_1d).min().values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for volatility and trailing stop ===
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
    
    warmup = max(50, lookback_1d, lookback_week + 1, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_1d[i]) or np.isnan(lowest_low_1d[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
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
                # Exit if price breaks below S1 (support 1) - mean reversion
                elif price < s1_aligned[i]:
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
                # Exit if price breaks above R1 (resistance 1) - mean reversion
                elif price > r1_aligned[i]:
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
            # Determine market bias relative to weekly pivot
            price_vs_pivot = price - weekly_pivot_aligned[i]
            
            # Long entry: price breaks above 1d Donchian high with bullish bias (above pivot)
            if (price > highest_high_1d[i] and 
                price_vs_pivot > 0):  # Above weekly pivot = bullish bias
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 1d Donchian low with bearish bias (below pivot)
            elif (price < lowest_low_1d[i] and 
                  price_vs_pivot < 0):  # Below weekly pivot = bearish bias
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