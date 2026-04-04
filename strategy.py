#!/usr/bin/env python3
"""
Experiment #3865: 12h Donchian(20) breakout + 1d EMA(200) trend filter + volume confirmation + chop regime filter
HYPOTHESIS: 12h Donchian breakouts aligned with 1d EMA(200) trend capture institutional participation in major moves.
Volume > 1.3x MA(30) confirms breakout strength. Chop regime filter (CHOP > 61.8) avoids whipsaw in ranging markets.
Discrete sizing (0.25) limits fee drag. ATR(14) trailing stop (2.0x) manages risk.
Target: 75-150 trades over 4 years (19-37/year) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3865_12h_donchian20_1d_ema_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA(200) and Chop regime filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(200) on 1d close
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate Chop regime filter on 1d data
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        tr1 = high_arr[1:] - low_arr[1:]
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        hh = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        ll = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        
        sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
        highest_high = pd.Series(hh).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(ll).rolling(window=period, min_periods=period).min().values
        
        chop = np.full(len(close_arr), np.nan)
        mask = (sum_atr > 0) & (highest_high > lowest_low)
        chop[mask] = 100 * np.log10(sum_atr[mask] / (highest_high[mask] - lowest_low[mask])) / np.log10(period)
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Align HTF indicators to 12h timeframe (shifted by 1 for completed 1d bar)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 12h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 12h Indicators: Volume MA(30) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[30:] = volume[30:] / vol_ma[30:]
    
    # === 12h Indicators: ATR(14) for volatility and trailing stop ===
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
    
    warmup = max(lookback_dc + 1, 30, 200)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Require volume spike (> 1.3x average) to filter noise
        volume_spike = vol_ratio[i] > 1.3
        
        # Chop regime filter: only trade when NOT choppy (CHOP < 61.8 = trending)
        not_choppy = chop_1d_aligned[i] < 61.8
        
        if volume_spike and not_choppy:
            # Determine trend direction from 1d EMA(200)
            trend_up = price > ema_200_1d_aligned[i]
            trend_down = price < ema_200_1d_aligned[i]
            
            # Long entry: price > EMA200 + Donchian upper breakout + volume
            long_signal = trend_up and price > highest_high[i-1]
            
            # Short entry: price < EMA200 + Donchian lower breakdown + volume
            short_signal = trend_down and price < lowest_low[i-1]
            
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