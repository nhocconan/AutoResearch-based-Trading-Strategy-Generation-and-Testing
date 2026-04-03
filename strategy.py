#!/usr/bin/env python3
"""
Experiment #152: 12h Camarilla Pivot + 1d Volume Spike + Chop Regime Filter + ATR Stoploss

HYPOTHESIS: Camarilla pivot levels on 12h timeframe act as strong support/resistance zones.
When price touches these levels with volume confirmation (>1.5x average) and the market 
is in a choppy regime (CHOP > 50), mean reversion trades have high probability.
In trending regimes (CHOP < 50), we fade the touch only if aligned with 1d trend.
This strategy targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
to minimize fee drag while capturing high-probability reversals and continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close for trend filter
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for chop regime (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Chopiness Index(14) on 1w
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr_1w = np.zeros(len(close_1w))
        tr_1w[0] = high_1w[0] - low_1w[0]
        for i in range(1, len(close_1w)):
            tr_1w[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
        
        # Sum of TR over 14 periods
        sum_tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
        ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
        
        # Chopiness Index
        chop_1w = np.zeros(len(close_1w))
        for i in range(len(close_1w)):
            if sum_tr_14[i] > 0 and hh_14[i] > ll_14[i]:
                chop_1w[i] = 100 * np.log10(sum_tr_14[i] / (hh_14[i] - ll_14[i])) / np.log10(14)
            else:
                chop_1w[i] = 50.0  # Neutral
        
        chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    else:
        chop_1w_aligned = np.full(n, 50.0)  # Default to neutral
    
    # === 12h Indicators: Camarilla Pivot Levels (based on previous 12h bar) ===
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Previous bar's OHLC
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        
        # Typical price for pivot
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        
        # Camarilla levels
        camarilla_h3[i] = pivot + range_val * 1.1 / 4
        camarilla_l3[i] = pivot - range_val * 1.1 / 4
        camarilla_h4[i] = pivot + range_val * 1.1 / 2
        camarilla_l4[i] = pivot - range_val * 1.1 / 2
    
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
    
    warmup = 50  # Ensure enough data for HTF EMA50 and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Chopiness Index from 1w ---
        choppy_market = chop_1w_aligned[i] > 50  # CHOP > 50 = choppy/range
        trending_market = chop_1w_aligned[i] <= 50  # CHOP <= 50 = trending
        
        # --- Trend Filter: 1d EMA50 ---
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Price Proximity to Camarilla Levels (within 0.1%) ---
        proximity_h3 = abs(close[i] - camarilla_h3[i]) / close[i] < 0.001
        proximity_l3 = abs(close[i] - camarilla_l3[i]) / close[i] < 0.001
        proximity_h4 = abs(close[i] - camarilla_h4[i]) / close[i] < 0.001
        proximity_l4 = abs(close[i] - camarilla_l4[i]) / close[i] < 0.001
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # In choppy market: mean reversion at H3/L3 levels
        if choppy_market and volume_spike:
            # Long: price near L3 with rejection (close > open) OR touching L3
            long_chop = (proximity_l3 or (close[i] < camarilla_l3[i] and close[i] > open[i])) and volume_spike
            # Short: price near H3 with rejection (close < open) OR touching H3
            short_chop = (proximity_h3 or (close[i] > camarilla_h3[i] and close[i] < open[i])) and volume_spike
            
            if long_chop:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif short_chop:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
        
        # In trending market: continuation break of H4/L4 with trend alignment
        elif trending_market and volume_spike:
            # Long: break above H4 with price above 1d EMA50
            long_trend = (close[i] > camarilla_h4[i]) and price_above_1d_ema and volume_spike
            # Short: break below L4 with price below 1d EMA50
            short_trend = (close[i] < camarilla_l4[i]) and price_below_1d_ema and volume_spike
            
            if long_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif short_trend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
        
        # Default: no signal
        if not in_position:
            signals[i] = 0.0
    
    return signals