#!/usr/bin/env python3
"""
Experiment #3850: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: 1d Donchian breakouts aligned with 1-week HMA trend capture medium-term momentum.
Volume > 1.5x MA(20) confirms institutional participation. Works in bull/bear: In uptrend,
buy upper band breaks; in downtrend, short lower band breaks. In ranging markets (price
between HMA bands), no entries to avoid whipsaw. Discrete sizing (0.25) limits fee drag.
ATR(14) trailing stop (2.0x) manages risk. Target: 30-100 trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3850_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate HMA(21) on 1w
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half_period, adjust=False).mean().values
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw = 2 * wma2 - wma1
        hma_vals = pd.Series(raw).ewm(span=sqrt_period, adjust=False).mean().values
        return hma_vals
    
    hma_21 = hma(close_1w, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # === 1d Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
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
                if price > lowest_since_entry + 2.0 * atr[i]:
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
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine market regime based on price vs 1w HMA
            # If price > HMA: bullish bias -> long breakouts
            # If price < HMA: bearish bias -> short breakdowns
            # If price near HMA: ranging -> no entries (avoid whipsaw)
            
            hma_val = hma_21_aligned[i]
            
            # Long entry conditions
            long_signal = False
            if price > hma_val and price > highest_high[i-1]:
                # Bullish breakout above upper Donchian in uptrend
                long_signal = True
            
            # Short entry conditions
            short_signal = False
            if price < hma_val and price < lowest_low[i-1]:
                # Bearish breakdown below lower Donchian in downtrend
                short_signal = True
            
            if long_signal and not short_signal:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_signal and not long_signal:
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