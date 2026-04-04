#!/usr/bin/env python3
"""
Experiment #4199: 6h Donchian(20) breakout + 12h/1d HTF regime filter + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture intermediate-term momentum when aligned with 12h/1d trend regime
(using EMA cross and price > SMA200) and confirmed by volume (>2.0x average). Uses discrete position sizing
(0.25) targeting 75-150 total trades over 4 years (19-38/year). Works in bull/bear via HTF regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4199_6h_donchian20_12h_1d_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Precompute HTF: 12h and 1d data for regime filter ===
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # 12h EMA(50) for intermediate trend
    if len(df_12h) >= 50:
        close_12h = df_12h['close'].values
        ema_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    else:
        ema_12h_aligned = np.full(n, np.nan)
    
    # 1d EMA(50) and SMA(200) for long-term regime
    if len(df_1d) >= 200:
        close_1d = df_1d['close'].values
        ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        sma_1d_200 = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
        sma_1d_200_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_200)
    else:
        ema_1d_aligned = np.full(n, np.nan)
        sma_1d_200_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20, 50, 200)  # Donchian, vol MA, EMA12h, SMA1d
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(sma_1d_200_aligned[i])):
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
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        if volume_confirm:
            # Regime filter: 12h EMA > 1d EMA = bullish regime, < = bearish regime
            # Plus price > 1d SMA200 for bullish, price < 1d SMA200 for bearish
            bullish_regime = (ema_12h_aligned[i] > ema_1d_aligned[i]) and (price > sma_1d_200_aligned[i])
            bearish_regime = (ema_12h_aligned[i] < ema_1d_aligned[i]) and (price < sma_1d_200_aligned[i])
            
            # Donchian breakout conditions
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # Long conditions: Donchian breakout up + bullish regime + volume confirmation
            long_entry = breakout_up and bullish_regime
            
            # Short conditions: Donchian breakout down + bearish regime + volume confirmation
            short_entry = breakout_dn and bearish_regime
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
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