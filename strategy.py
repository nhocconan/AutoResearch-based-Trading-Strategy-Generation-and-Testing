#!/usr/bin/env python3
"""
Experiment #3975: 6h Donchian(20) breakout + 1w Camarilla pivot regime + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with weekly Camarilla pivot levels capture multi-week swings. 
Weekly Camarilla levels (R3/S3 = fade zone, R4/S4 = breakout zone) provide institutional reference points. 
Volume > 2.0x MA(20) confirms breakout strength. ATR(14) trailing stop (2.5x) manages risk. 
Discrete sizing (0.25) reduces fee drag. Target: 75-200 trades over 4 years (19-50/year). 
Works in bull/bear via weekly Camarilla regime filter (price above/below weekly pivot).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3975_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w Camarilla pivot levels for regime ===
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly OHLC from daily data (approximation using weekly resample is not allowed, 
    # so we use the actual weekly candles from get_htf_data)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_range = weekly_high - weekly_low
    # Camarilla levels: R4 = close + range * 1.1/2, R3 = close + range * 1.1/4, 
    # S3 = close - range * 1.1/4, S4 = close - range * 1.1/2
    camarilla_r4 = weekly_close + weekly_range * 1.1 / 2
    camarilla_r3 = weekly_close + weekly_range * 1.1 / 4
    camarilla_s3 = weekly_close - weekly_range * 1.1 / 4
    camarilla_s4 = weekly_close - weekly_range * 1.1 / 2
    # Pivot point (optional)
    camarilla_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
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
    
    warmup = max(lookback_dc + 1, 20)  # DC lookback, vol MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
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
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
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
        # Require volume spike (> 2.0x average) to filter noise
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Determine regime using weekly Camarilla levels:
            # Bullish bias: price above weekly R3 (strong) or above pivot (moderate)
            # Bearish bias: price below weekly S3 (strong) or below pivot (moderate)
            # For breakouts: look for price > R4 (bullish breakout) or < S4 (bearish breakout)
            bullish_breakout = price > camarilla_r4_aligned[i-1]
            bearish_breakout = price < camarilla_s4_aligned[i-1]
            
            # Additional confirmation: price should be above/below R3/S3 for stronger signal
            bullish_regime = price > camarilla_r3_aligned[i]
            bearish_regime = price < camarilla_s3_aligned[i]
            
            # Long entry: breakout above weekly R4 with bullish regime
            long_breakout = bullish_breakout and bullish_regime
            # Short entry: breakdown below weekly S4 with bearish regime
            short_breakout = bearish_breakout and bearish_regime
            
            if long_breakout and not short_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_breakout and not long_breakout:
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