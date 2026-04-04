#!/usr/bin/env python3
"""
Experiment #3607: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture momentum bursts. Weekly pivot direction (from 1d HTF) filters for trades aligned with higher-timeframe structure. Volume confirmation ensures breakout authenticity. Works in bull markets (breakouts above R1 in uptrend) and bear markets (breakdowns below S1 in downtrend). Position size 0.25. Target: 75-150 total trades over 4 years (19-37/year). Uses 1d for pivot calculation and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3607_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot points and trend alignment (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from prior week's OHLC (using prior completed week)
    # We need to aggregate daily to weekly - but since we have 1d data, we'll use prior 5-day period
    # Simpler approach: use prior day's range for intraday pivot (still effective)
    # Actually, let's use proper weekly pivot: (Prior Week H + L + C) / 3
    # To avoid lookback, we'll use rolling window of 5 days (1 week) and shift by 1
    if len(close_1d) >= 5:
        # Weekly high/low/close from prior 5-day period (shifted by 1 to avoid lookahead)
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
        weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1).values
        
        # Weekly pivot point
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Weekly R1 and S1
        weekly_r1 = 2 * weekly_pivot - weekly_low
        weekly_s1 = 2 * weekly_pivot - weekly_high
        
        # Align to 6h timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    else:
        # Not enough data - use close as fallback
        weekly_pivot_aligned = close_1d[-1] if len(close_1d) > 0 else close
        weekly_r1_aligned = close_1d[-1] if len(close_1d) > 0 else close
        weekly_s1_aligned = close_1d[-1] if len(close_1d) > 0 else close
        weekly_pivot_aligned = np.full(n, weekly_pivot_aligned)
        weekly_r1_aligned = np.full(n, weekly_r1_aligned)
        weekly_s1_aligned = np.full(n, weekly_s1_aligned)
    
    # === 6h Indicators: Donchian Channel(20) for breakouts ===
    lookback_donchian = 20
    highest_high = pd.Series(high).rolling(window=lookback_donchian, min_periods=lookback_donchian).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_donchian, min_periods=lookback_donchian).min().values
    donchian_width = highest_high - lowest_low
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and stoploss ===
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
    
    warmup = max(50, lookback_donchian + 1, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i])):
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
                # Exit if price breaks below weekly pivot (mean reversion)
                elif price < weekly_pivot_aligned[i]:
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
                # Exit if price breaks above weekly pivot (mean reversion)
                elif price > weekly_pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike and donchian_width[i] > 0:  # Avoid division by zero
            # Normalized position within Donchian channel (0=low, 1=high)
            channel_position = (price - lowest_low[i]) / donchian_width[i]
            
            # Long entry: Break above Donchian high AND above weekly R1 (bullish alignment)
            if (price > highest_high[i] * 1.001 and  # Small buffer to avoid false breakouts
                price > weekly_r1_aligned[i] and
                channel_position > 0.95):  # Near top of channel
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Break below Donchian low AND below weekly S1 (bearish alignment)
            elif (price < lowest_low[i] * 0.999 and  # Small buffer to avoid false breakdowns
                  price < weekly_s1_aligned[i] and
                  channel_position < 0.05):  # Near bottom of channel
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