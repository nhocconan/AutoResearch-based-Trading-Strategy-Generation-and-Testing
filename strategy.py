#!/usr/bin/env python3
"""
Experiment #3814: 1h Donchian(20) breakout + 4h volume confirmation + 1d trend filter
HYPOTHESIS: 1h Donchian breakouts capture short-term swings with 4h volume (>1.5x) confirming participation and 1d EMA(50) filter ensuring trend alignment. Works in bull markets (breakouts above resistance in uptrend) and bear markets (breakdowns below support in downtrend). Session filter (08-20 UTC) reduces noise trades. Discrete position sizing (0.20) minimizes fee drag. Target: 60-150 total trades over 4 years = 15-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3814_1h_donchian20_4h_vol_1d_ema_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for volume confirmation (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = np.ones(len(volume_4h))
    vol_ratio_4h[20:] = volume_4h[20:] / vol_ma_4h[20:]
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    warmup = max(lookback_dc + 1, 20, 50)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                # Calculate ATR(14) for trailing stop
                if i >= 14:
                    prev_close = close[i-1]
                    tr = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
                    atr_14 = np.mean([tr] + [max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1])) 
                                          for j in range(max(0, i-13), i)]) if i >= 1 else tr
                    if price < highest_since_entry - 2.0 * atr_14:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                # Exit if price breaks below Donchian lower band (trend reversal)
                elif price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if i >= 14:
                    prev_close = close[i-1]
                    tr = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
                    atr_14 = np.mean([tr] + [max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1])) 
                                          for j in range(max(0, i-13), i)]) if i >= 1 else tr
                    if price > lowest_since_entry + 2.0 * atr_14:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                # Exit if price breaks above Donchian upper band (trend reversal)
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) on 4h AND trend alignment on 1d
        volume_spike = vol_ratio_4h_aligned[i] > 1.5
        # Uptrend: price > EMA50, Downtrend: price < EMA50
        uptrend = price > ema_50_1d_aligned[i]
        downtrend = price < ema_50_1d_aligned[i]
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND uptrend
            if (price > highest_high[i-1] and uptrend):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND downtrend
            elif (price < lowest_low[i-1] and downtrend):
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