#!/usr/bin/env python3
"""
Experiment #203: 4h Donchian Breakout + HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: Donchian(20) breakouts on 4h with HMA(21) trend filter and volume confirmation (>1.5x average) capture institutional breakout moves. Works in bull/bear regimes by only trading in direction of higher timeframe trend (12h). ATR-based stoploss (2x) manages risk. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_203_4h_donchian_hma_volume_12h_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h close for trend filter
    def hma(series, period):
        """Hull Moving Average"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma1 = pd.Series(series).ewm(span=half_period, adjust=False).mean()
        wma2 = pd.Series(series).ewm(span=period, adjust=False).mean()
        raw_hma = 2 * wma1 - wma2
        hma_result = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
        return hma_result.values
    
    close_12h = df_12h['close'].values
    hma_21_12h = hma(close_12h, 21)
    trend_up_12h = close_12h > hma_21_12h
    trend_down_12h = close_12h < hma_21_12h
    
    # Align to 4h timeframe
    trend_up_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_up_12h)
    trend_down_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_down_12h)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(trend_up_12h_aligned[i]) or np.isnan(trend_down_12h_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price reaches Donchian upper band with volume (continuation) or reverse
                if price >= highest_20[i] and volume_spike:
                    # Continue the trend
                    signals[i] = SIZE
                else:
                    # Take profit/exit
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price reaches Donchian lower band with volume (continuation) or reverse
                if price <= lowest_20[i] and volume_spike:
                    # Continue the trend
                    signals[i] = -SIZE
                else:
                    # Take profit/exit
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper band with volume in uptrend
        if (price > highest_20[i] and 
            trend_up_12h_aligned[i] and 
            volume_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below Donchian lower band with volume in downtrend
        elif (price < lowest_20[i] and 
              trend_down_12h_aligned[i] and 
              volume_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>