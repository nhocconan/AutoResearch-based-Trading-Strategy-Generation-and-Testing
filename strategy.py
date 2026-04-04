#!/usr/bin/env python3
"""
Experiment #5079: 6h Elder Ray + ADX Regime + Volume Spike
HYPOTHESIS: On 6h timeframe, Elder Ray (Bull/Bear Power) combined with ADX regime filter captures strong trending moves while avoiding chop. Bull Power > 0 and Bear Power < 0 with ADX > 25 indicates strong trend. Volume > 1.5x average confirms participation. ATR(14) trailing stop (2.0x) manages risk. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in both bull (strong upward thrusts) and bear (strong downward thrusts) markets by filtering for genuine momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5079_6h_elder_ray_adx_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === 6h Indicators: EMA(13) for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === 6h Indicators: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) ===
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 6h Indicators: ADX(14) for regime filter ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_ma = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (np.abs(di_plus) + np.abs(di_minus)) * 100
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(13, 14, 20, 14)  # EMA13, ADX, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Regime filter: ADX > 25 indicates trending market
        trending = adx[i] > 25
        
        # Elder Ray conditions:
        # Long: Bull Power > 0 (strong bullish momentum) AND Bear Power < 0 (no bearish pressure)
        # Short: Bear Power < 0 (strong bearish momentum) AND Bull Power > 0 (no bullish pressure)
        # Actually: For long, we want Bull Power > 0 AND Bear Power < 0 (both conditions show bullish bias)
        # For short, we want Bear Power < 0 AND Bull Power > 0 (both conditions show bearish bias)
        # Wait, that's the same. Let me reconsider:
        # Bull Power = High - EMA > 0 means bulls are pushing price above average
        # Bear Power = Low - EMA < 0 means bears are pushing price below average
        # For long: We want Bull Power > 0 (bulls in control) 
        # For short: We want Bear Power < 0 (bears in control)
        # But we also need to avoid chop: both shouldn't be near zero
        long_signal = (bull_power[i] > 0) and (bear_power[i] < 0)  # Bulls in, bears out
        short_signal = (bear_power[i] < 0) and (bull_power[i] > 0)  # Same condition - this is wrong
        
        # Correct Elder Ray interpretation:
        # Bull Power > 0 indicates bulls are stronger than the average
        # Bear Power < 0 indicates bears are stronger than the average
        # For a strong trend, we want one to be significantly positive/negative
        # Long: Bull Power > 0 AND Bear Power < some small threshold (lets say -0.1*price) 
        # Actually simpler: Long when Bull Power > 0 (bulls pushing up)
        # Short when Bear Power < 0 (bears pushing down)
        # But we need to avoid false signals in chop - hence ADX filter
        long_signal = bull_power[i] > 0
        short_signal = bear_power[i] < 0
        
        # Final entry conditions
        if long_signal and trending and vol_confirm:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_signal and trending and vol_confirm:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals