#!/usr/bin/env python3
"""
Experiment #139: 6h Camarilla Pivot + 12h Volume Regime + ATR Trend Filter

HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with 12h volume regime (high/low volume) and 12h ATR trend filter captures 
high-probability trades in both bull and bear markets. In low volume regimes, 
fade extreme Camarilla levels (R3/S3). In high volume regimes, breakout through 
R4/S4 with trend confirmation. Uses ATR-based stops for risk management. 
Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_139_6h_camarilla_12h_volregime_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivots, volume regime, ATR trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels from prior 12h bar (using prior bar's OHLC)
    # Camarilla: 
    # R4 = Close + (High - Low) * 1.1/2
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    # S4 = Close - (High - Low) * 1.1/2
    prior_close = df_12h['close'].values
    prior_high = df_12h['high'].values
    prior_low = df_12h['low'].values
    prior_range = prior_high - prior_low
    
    camarilla_r4 = prior_close + prior_range * 1.1 / 2.0
    camarilla_r3 = prior_close + prior_range * 1.1 / 4.0
    camarilla_s3 = prior_close - prior_range * 1.1 / 4.0
    camarilla_s4 = prior_close - prior_range * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # 12h Volume regime: high volume if current volume > 1.5x 24-period MA
    vol_ma_24 = pd.Series(df_12h['volume'].values).rolling(window=24, min_periods=24).mean().values
    vol_ratio_12h = df_12h['volume'].values / vol_ma_24
    vol_ratio_12h[vol_ma_24 == 0] = 1.0  # Avoid division by zero
    vol_regime_high = align_htf_to_ltf(prices, df_12h, vol_ratio_12h > 1.5)
    
    # 12h ATR trend filter: ATR(14) > ATR(50) indicates trending market
    tr_12h = np.zeros(len(df_12h))
    tr_12h[0] = df_12h['high'].iloc[0] - df_12h['low'].iloc[0]
    for i in range(1, len(df_12h)):
        tr_12h[i] = max(
            df_12h['high'].iloc[i] - df_12h['low'].iloc[i],
            abs(df_12h['high'].iloc[i] - df_12h['close'].iloc[i-1]),
            abs(df_12h['low'].iloc[i] - df_12h['close'].iloc[i-1])
        )
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_50_12h = pd.Series(tr_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    atr_trend = align_htf_to_ltf(prices, df_12h, atr_14_12h > atr_50_12h)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Ensure enough data for HTF calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Regime and Trend Filters (from 12h) ---
        high_volume_regime = vol_regime_high[i] if not np.isnan(vol_regime_high[i]) else False
        trending_market = atr_trend[i] if not np.isnan(atr_trend[i]) else False
        
        # --- Camarilla-Based Entry Logic ---
        long_signal = False
        short_signal = False
        
        if high_volume_regime and trending_market:
            # High volume + trending: breakout continuation at R4/S4
            if close[i] > camarilla_r4_aligned[i]:
                long_signal = True
            elif close[i] < camarilla_s4_aligned[i]:
                short_signal = True
        else:
            # Low volume or ranging: mean reversion at extreme levels (R3/S3)
            if close[i] < camarilla_s3_aligned[i]:
                long_signal = True  # Oversold bounce
            elif close[i] > camarilla_r3_aligned[i]:
                short_signal = True  # Overbought reversal
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if long_signal:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_signal:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals