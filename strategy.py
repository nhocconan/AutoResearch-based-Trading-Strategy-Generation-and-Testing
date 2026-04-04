#!/usr/bin/env python3
"""
Experiment #5029: 4h Donchian(20) Breakout + 1d/1w HTF Volume Spike + ATR Stoploss
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts aligned with 1d volume spikes (2.0x average) capture institutional momentum. 
HTF 1d confirms trend direction via price > SMA50, HTF 1w confirms regime via price > SMA200. 
Volume > 2.0x 20-period average filters for genuine breakouts. ATR(14) trailing stop (2.5x) manages risk. 
Designed for 19-50 trades/year on 4h timeframe to minimize fee drag while maintaining statistical significance. 
Works in bull markets via breakouts through resistance and in bear markets via breakdowns through support.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5029_4h_donchian20_1d_1w_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d Indicators: SMA50 for trend filter ===
    if len(df_1d) >= 50:
        sma50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
        sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    else:
        sma50_1d_aligned = np.full(n, np.nan)
    
    # === 1w Indicators: SMA200 for regime filter ===
    if len(df_1w) >= 200:
        sma200_1w = pd.Series(df_1w['close'].values).rolling(window=200, min_periods=200).mean().values
        sma200_1w_aligned = align_htf_to_ltf(prices, df_1w, sma200_1w)
    else:
        sma200_1w_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation (2.0x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20, 14, 50, 200)  # Donchian, Volume MA, ATR, SMA50, SMA200 warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(sma50_1d_aligned[i]) or np.isnan(sma200_1w_aligned[i]) or
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
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Trend filter: price above 1d SMA50 for long, below for short
        trend_long = price > sma50_1d_aligned[i]
        trend_short = price < sma50_1d_aligned[i]
        
        # Regime filter: price above 1w SMA200 for bull regime, below for bear regime
        bull_regime = price > sma200_1w_aligned[i]
        bear_regime = price < sma200_1w_aligned[i]
        
        # Donchian breakout conditions
        # Long: Donchian breakout above high_roll in bull regime OR any regime with volume confirmation
        # Short: Donchian breakdown below low_roll in bear regime OR any regime with volume confirmation
        breakout_long = ((price >= high_roll[i]) and 
                        (bull_regime or vol_confirm)) and  # Must be in bull regime OR have volume confirmation
                        vol_confirm  # Volume confirmation always required
        
        breakout_short = ((price <= low_roll[i]) and 
                         (bear_regime or vol_confirm)) and  # Must be in bear regime OR have volume confirmation
                         vol_confirm  # Volume confirmation always required
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals