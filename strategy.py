#!/usr/bin/env python3
"""
Experiment #3524: 1d Donchian Breakout + 1w HMA Trend + Volume Confirmation
HYPOTHESIS: 1d Donchian(20) breakouts with 1-week HMA(21) direction and volume confirmation capture medium-term momentum.
Weekly HMA provides the trend filter, Donchian breakout provides entry timing, volume confirms strength.
Position size 0.25. Target: 75-150 total trades over 4 years (19-38/year).
Uses 1d for primary timeframe and 1w for HTF trend filter.
Works in bull (breakout above Donchian high with bullish weekly trend) and bear (breakout below Donchian low with bearish weekly trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3524_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate HMA(21) on weekly data
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half_period, adjust=False).mean()
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
        raw_hma = 2 * wma2 - wma1
        hma_vals = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
        return hma_vals.values
    
    hma_21_1w = hma(close_1w, 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # === 1d Indicators: Donchian channels (20-period) for entry timing ===
    lookback_1d = 20
    highest_high_1d = pd.Series(high).rolling(window=lookback_1d, min_periods=lookback_1d).max().values
    lowest_low_1d = pd.Series(low).rolling(window=lookback_1d, min_periods=lookback_1d).min().values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for volatility and stoploss ===
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
    
    warmup = max(50, lookback_1d, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_1d[i]) or np.isnan(lowest_low_1d[i]) or
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Require volume spike (> 1.5x average) for confirmation
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine trend from weekly HMA
            weekly_trend = hma_21_1w_aligned[i]
            
            # Long entry: price breaks above 1d Donchian high with bullish weekly trend
            if (price > highest_high_1d[i] and 
                close[i] > weekly_trend):  # Price above weekly HMA = bullish bias
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 1d Donchian low with bearish weekly trend
            elif (price < lowest_low_1d[i] and 
                  close[i] < weekly_trend):  # Price below weekly HMA = bearish bias
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