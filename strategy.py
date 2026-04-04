#!/usr/bin/env python3
"""
Experiment #5051: 6h Elder Ray + 1d ADX Regime Filter + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Elder Ray (Bull Power/Bear Power) combined with 1d ADX regime filter captures trend strength with institutional confirmation. ADX > 25 on 1d filters for trending regimes (works in both bull/bear markets), while Elder Ray identifies exhaustion points. Volume > 1.5x average confirms participation. ATR-based trailing stop (2.0x) manages risk. Targets 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets via long signals when Bull Power > 0 and ADX trending, and in bear markets via short signals when Bear Power < 0 and ADX trending.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5051_6h_elder_ray_1d_adx_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: ADX(14) for regime filter ===
    if len(df_1d) >= 14:
        # True Range
        tr1 = df_1d['high'][1:] - df_1d['low'][1:]
        tr2 = np.abs(df_1d['high'][1:] - df_1d['close'][:-1])
        tr3 = np.abs(df_1d['low'][1:] - df_1d['close'][:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((df_1d['high'][1:] - df_1d['high'][:-1]) > (df_1d['low'][:-1] - df_1d['low'][1:]),
                           np.maximum(df_1d['high'][1:] - df_1d['high'][:-1], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.where((df_1d['low'][:-1] - df_1d['low'][1:]) > (df_1d['high'][1:] - df_1d['high'][:-1]),
                            np.maximum(df_1d['low'][:-1] - df_1d['low'][1:], 0), 0)
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed TR, DM+
        tr_smooth = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # DI+ and DI-
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Align to 6h timeframe
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Elder Ray (Bull Power/Bear Power) ===
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
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
    
    warmup = max(13, 20, 14, 14)  # EMA13, Volume MA, ATR, ADX warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        trending_regime = adx_aligned[i] > 25
        
        # Elder Ray conditions
        # Long: Bull Power > 0 (bulls in control) + trending regime + volume
        # Short: Bear Power < 0 (bears in control) + trending regime + volume
        long_signal = (bull_power[i] > 0) and trending_regime and vol_confirm
        short_signal = (bear_power[i] < 0) and trending_regime and vol_confirm
        
        # Final entry conditions
        if long_signal:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_signal:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals