#!/usr/bin/env python3
"""
Experiment #1522: 12h Donchian(20) Breakout + 1d/1w Trend + Volume + Chop Filter
HYPOTHESIS: 12h Donchian breakouts with 1d/1w trend alignment, volume confirmation (>1.5x average), and chop filter (Choppiness Index > 61.8 = ranging) capture medium-term swings while avoiding whipsaws in strong trends. Position size fixed at 0.25 to balance risk and return. Target: 75-150 total trades over 4 years (19-37/year) by using tight entry conditions and multi-timeframe confluence. Works in bull (trend breakouts) and bear (mean reversion in chop regimes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1522_12h_donchian20_1d1w_trend_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for filter
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 12h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Choppiness Index(14) for regime filter ===
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.log10(14) / (max_high - min_low))
    chop = np.where((max_high - min_low) != 0, chop, 50)  # avoid division by zero
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(trend_1w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 08-20 session filter
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require both 1d and 1w trend alignment
        trend_following = (trend_1d_aligned[i] > 0) and (trend_1w_aligned[i] > 0)
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        # Chop filter: only enter when chop > 61.8 (ranging market) for mean reversion
        # or when chop < 38.2 (trending) for trend following - we use both regimes
        chop_regime = (chop[i] > 61.8) or (chop[i] < 38.2)
        
        # Session filter: only trade during active hours
        if trend_following and volume_spike and chop_regime and in_session:
            # Breakout: price breaks above upper band OR below lower band
            if price > donch_high[i] and trend_1d_aligned[i] > 0 and trend_1w_aligned[i] > 0:  # Uptrend breakout
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donch_low[i] and trend_1d_aligned[i] < 0 and trend_1w_aligned[i] < 0:  # Downtrend breakdown
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals