#!/usr/bin/env python3
"""
Experiment #119: 6h Camarilla Pivot + Volume Spike + Regime Filter

HYPOTHESIS: Camarilla pivot levels on 1d timeframe provide significant support/resistance zones. 
Breakouts above R4 or below S4 with volume confirmation indicate strong institutional interest, 
while fade trades at R3/S3 with volume exhaustion capture mean reversion in ranging markets. 
The 6h timeframe balances trade frequency and signal quality, targeting 12-37 trades/year. 
Works in both bull/bear regimes by switching between breakout and mean-reversion logic based 
on volatility regime (choppiness index).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d
    if len(df_1d) >= 2:
        # Use previous day's OHLC for today's Camarilla levels
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        
        # Camarilla levels
        r3 = pivot + (range_val * 1.1 / 2)
        r4 = pivot + (range_val * 1.1)
        s3 = pivot - (range_val * 1.1 / 2)
        s4 = pivot - (range_val * 1.1)
        
        # Align to 6h timeframe (shifted by 1 day for lookback safety)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        r3_aligned = r4_aligned = s3_aligned = s4_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for Choppiness Index regime filter (Call ONCE before loop) ===
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high_1d[0] - low_1d[0]  # First period
        
        # Sum of TR over 14 periods
        tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        
        # MaxHigh - MinLow over 14 periods
        max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        range_maxmin = max_high - min_low
        
        # Choppiness Index: CHI = 100 * log10(sumTR/(range_maxmin)) / log10(14)
        chop = np.zeros(len(close_1d))
        mask = (tr_sum > 0) & (range_maxmin > 0)
        chop[mask] = 100 * np.log10(tr_sum[mask] / range_maxmin[mask]) / np.log10(14)
        chop[:13] = 50.0  # Neutral for warmup
        
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    else:
        chop_aligned = np.full(n, 50.0)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Detection: Choppiness Index ---
        # CHOP > 61.8 = ranging market (mean revert)
        # CHOP < 38.2 = trending market (breakout)
        is_ranging = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using 6h data
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit logic based on regime
                if is_ranging and close[i] >= s3_aligned[i]:  # Mean revert to S3 in ranging
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                elif is_trending and close[i] <= r3_aligned[i]:  # Trail to R3 in trending
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit logic based on regime
                if is_ranging and close[i] <= r3_aligned[i]:  # Mean revert to R3 in ranging
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                elif is_trending and close[i] >= s3_aligned[i]:  # Trail to S3 in trending
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if is_ranging:
            # Ranging market: Mean reversion at R3/S3 with volume exhaustion
            # Look for volume NOT spiking (exhaustion) near extremes
            volume_normal = vol_ratio_1d_aligned[i] < 1.2  # Not spiking
            
            # Long: Price at S3 with volume exhaustion
            long_condition = (
                close[i] <= s3_aligned[i] * 1.005 and  # Within 0.5% of S3
                volume_normal and
                close[i] > close[i-1]  # Price starting to rise
            )
            
            # Short: Price at R3 with volume exhaustion
            short_condition = (
                close[i] >= r3_aligned[i] * 0.995 and  # Within 0.5% of R3
                volume_normal and
                close[i] < close[i-1]  # Price starting to fall
            )
        else:
            # Trending market: Breakout continuation at R4/S4 with volume confirmation
            # Long: Break above R4 with volume spike
            long_condition = (
                close[i] > r4_aligned[i] and
                volume_spike and
                close[i] > open[i]  # Bullish close
            )
            
            # Short: Break below S4 with volume spike
            short_condition = (
                close[i] < s4_aligned[i] and
                volume_spike and
                close[i] < open[i]  # Bearish close
            )
        
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