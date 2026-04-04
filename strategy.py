#!/usr/bin/env python3
"""
Experiment #3490: 1d Donchian Breakout + 1w EMA Trend Filter + Volume Spike + ATR Stop
HYPOTHESIS: 1d Donchian(20) breakouts with volume confirmation and 1w EMA trend alignment capture medium-term momentum while avoiding whipsaws. Uses 1w for trend direction, 1d for structure, and volume for confirmation. Works in bull markets (trend continuation) and bear markets (mean reversion from extremes via price channels). Target: 50-150 total trades over 4 years (12-38/year). Position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3490_1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === HTF: 1w data for EMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === 1d Indicators: Donchian channels (20-period) for structure ===
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
    
    warmup = max(50, lookback_1d, 20, 14, 50)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high_1d[i]) or np.isnan(lowest_low_1d[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
                # Exit if price re-enters 1d Donchian channel (mean reversion)
                elif price <= highest_high_1d[i]:
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
                # Exit if price re-enters 1d Donchian channel (mean reversion)
                elif price >= lowest_low_1d[i]:
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
            # 1d Donchian structure: only long above 1d highest, short below 1d lowest
            price_vs_1d_high = price - highest_high_1d[i]
            price_vs_1d_low = price - lowest_low_1d[i]
            
            # 1w EMA trend filter: only long above EMA, short below EMA
            price_vs_ema = price - ema_1w_aligned[i]
            
            # Long entry: price breaks above 1d Donchian high with bullish 1w trend
            if (price > highest_high_1d[i] and 
                price_vs_1d_high > 0 and 
                price_vs_ema > 0):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 1d Donchian low with bearish 1w trend
            elif (price < lowest_low_1d[i] and 
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