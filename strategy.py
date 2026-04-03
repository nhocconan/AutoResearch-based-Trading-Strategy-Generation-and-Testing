#!/usr/bin/env python3
"""
Experiment #148: 12h Donchian(20) Breakout + 1w Trend Filter + Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts on 12h timeframe, filtered by 1w trend (price > EMA50) 
and confirmed by volume spikes (>2x average), capture strong momentum moves in both bull 
and bear markets. The Donchian structure provides objective breakout levels, the 1w EMA50 
filter ensures alignment with higher timeframe trend (avoiding counter-trend trades), and 
volume confirmation filters out false breakouts. Targets 12-37 trades/year on 12h timeframe 
(50-150 total over 4 years) to minimize fee drag while capturing high-probability trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w close
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for regime filter (choppiness) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        tr_1d = np.zeros(len(close_1d))
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        plus_dm_1d = np.zeros(len(close_1d))
        minus_dm_1d = np.zeros(len(close_1d))
        for i in range(1, len(close_1d)):
            plus_dm_1d[i] = max(high_1d[i] - high_1d[i-1], 0)
            minus_dm_1d[i] = max(low_1d[i-1] - low_1d[i], 0)
        atr_14_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        plus_di_14_1d = 100 * pd.Series(plus_dm_1d).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_14_1d
        minus_di_14_1d = 100 * pd.Series(minus_dm_1d).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_14_1d
        dx_14_1d = 100 * np.abs(plus_di_14_1d - minus_di_14_1d) / (plus_di_14_1d + minus_di_14_1d)
        adx_14_1d = pd.Series(dx_14_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    else:
        adx_14_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel (20) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_m = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # === 12h Indicators: ATR(14) for stoploss and volume average ===
    atr_14 = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 200  # Ensure enough data for HTF EMA50 and ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(adx_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of 1w EMA50 ---
        price_above_1w_ema = close[i] > ema_50_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_50_1w_aligned[i]
        
        # --- Regime Filter: Only trade when trending (ADX > 25) ---
        trending = adx_14_1d_aligned[i] > 25
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] < donchian_m[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] > donchian_m[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + volume spike + price above 1w EMA50 + trending
        long_condition = breakout_up and volume_spike and price_above_1w_ema and trending
        
        # Short: Donchian breakout down + volume spike + price below 1w EMA50 + trending
        short_condition = breakout_down and volume_spike and price_below_1w_ema and trending
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>