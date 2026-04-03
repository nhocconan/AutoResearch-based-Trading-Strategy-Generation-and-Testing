#!/usr/bin/env python3
"""
Experiment #172: 12h Camarilla Pivot + Volume Spike + Chop Filter (1d/1w HTF)

HYPOTHESIS: 12h Camarilla pivot levels (L3, L4, H3, H4) act as strong support/resistance.
Long when price touches L3/L4 with volume spike (>2x) and choppy regime (CHOP>61.8).
Short when price touches H3/H4 with volume spike and choppy regime.
1d/1w HTF filters ensure alignment with higher timeframe structure.
ATR-based stoploss manages risk. Targets 12-37 trades/year (50-150 total over 4 years)
to minimize fee drag while capturing high-probability mean-reversion moves in ranging markets
and avoiding strong trends via chop filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_chop_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d and 1w data for regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d ADX(14) for trend strength (avoid strong trends)
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
        tr_1d[0] = tr1[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                           np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
        dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                            np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        tr_14 = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_14 = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_14 = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # DI and DX
        di_plus = 100 * dm_plus_14 / tr_14
        di_minus = 100 * dm_minus_14 / tr_14
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx_1d = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    else:
        adx_1d_aligned = np.full(n, 50.0)  # Neutral if insufficient data
    
    # Calculate 1w chopiness index (CHOP) for regime detection
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr1w = high_1w - low_1w
        tr2w = np.abs(high_1w - np.roll(close_1w, 1))
        tr3w = np.abs(low_1w - np.roll(close_1w, 1))
        tr_1w = np.maximum(tr1w, np.maximum(tr2w, tr3w))
        tr_1w[0] = tr1w[0]
        
        # Sum of TR over 14 periods
        tr_sum_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
        ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
        
        # Chopiness Index: CHOP = 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
        # Avoid division by zero
        range_14 = hh_14 - ll_14
        chop_1w = np.zeros_like(tr_sum_14, dtype=np.float64)
        mask = (range_14 > 0) & (~np.isnan(tr_sum_14)) & (~np.isnan(range_14))
        chop_1w[mask] = 100 * np.log10(tr_sum_14[mask] / range_14[mask]) / np.log10(14)
        chop_1w[~mask] = 50.0  # Neutral when range is zero
        
        chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    else:
        chop_1w_aligned = np.full(n, 50.0)  # Neutral if insufficient data
    
    # === 12h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Camarilla levels calculated from previous 12h bar's high, low, close
    # We need to shift the calculation by 1 bar to avoid look-ahead
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_close = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous bar's OHLC to calculate today's levels (no look-ahead)
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        
        # Camarilla equations
        range_ = phigh - plow
        camarilla_close[i] = pclose  # Midpoint reference
        camarilla_h3[i] = pclose + range_ * 1.1 / 4
        camarilla_l3[i] = pclose - range_ * 1.1 / 4
        camarilla_h4[i] = pclose + range_ * 1.1 / 2
        camarilla_l4[i] = pclose - range_ * 1.1 / 2
    
    # === 12h Indicators: ATR(14) for stoploss ===
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
    
    warmup = 200  # Ensure enough data for HTF indicators and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filters ---
        # Avoid strong trends (ADX > 25) - we want ranging markets for mean reversion
        not_strong_trend = adx_1d_aligned[i] < 25
        # Choppy regime (CHOP > 61.8) - favors mean reversion
        choppy_regime = chop_1w_aligned[i] > 61.8
        regime_filter = not_strong_trend and choppy_regime
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Price Levels ---
        price = close[i]
        near_l3 = abs(price - camarilla_l3[i]) / camarilla_l3[i] < 0.002  # Within 0.2%
        near_l4 = abs(price - camarilla_l4[i]) / camarilla_l4[i] < 0.002
        near_h3 = abs(price - camarilla_h3[i]) / camarilla_h3[i] < 0.002
        near_h4 = abs(price - camarilla_h4[i]) / camarilla_h4[i] < 0.002
        
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price near L3/L4 + volume spike + choppy regime (mean reversion up)
        long_condition = (near_l3 or near_l4) and volume_spike and regime_filter
        
        # Short: Price near H3/H4 + volume spike + choppy regime (mean reversion down)
        short_condition = (near_h3 or near_h4) and volume_spike and regime_filter
        
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