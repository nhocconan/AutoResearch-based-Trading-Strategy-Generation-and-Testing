#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v3
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation. 
Designed for low trade frequency (~25-40/year) to minimize fee drag. Uses 4h primary timeframe 
with 1d HTF for trend and volume context. Works in both bull/bear markets by requiring 
confluence of price structure, trend alignment, and volatility expansion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === ATR for stoploss (14-period on 4h) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Calculate Camarilla pivot levels from previous 1d bar ===
    # We need previous day's OHLC for today's Camarilla levels
    # Since df_1d contains completed daily bars, we can use it directly
    # For 4h bar i, we use the Camarilla levels from the most recent completed 1d bar
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        atr_val = atr_14[i]
        
        # Get previous completed 1d bar for Camarilla calculation
        # Find the index of the most recent completed 1d bar
        # We'll use a simple approach: since we have aligned arrays, 
        # we can use the HTF data directly with proper indexing
        
        # For simplicity in this implementation, we'll calculate Camarilla levels
        # using the previous day's data from df_1d
        # We need to map the 4h bar to the appropriate 1d bar
        
        # Calculate Camarilla levels from previous 1d bar
        # We'll use a rolling window approach on the 1d data and align it
        if i >= 96:  # Need at least 4 days of 4h data (4*24=96) to have meaningful 1d alignment
            # Get the index in df_1d that corresponds to the 1d bar ending at or before current 4h bar
            # Since we have 24 4h bars per 1d bar, we can use integer division
            idx_1d = i // 24
            if idx_1d > 0 and idx_1d < len(df_1d):
                # Use previous day's OHLC (idx_1d-1) to avoid look-ahead
                prev_idx = idx_1d - 1
                if prev_idx >= 0:
                    prev_high = df_1d['high'].iloc[prev_idx]
                    prev_low = df_1d['low'].iloc[prev_idx]
                    prev_close = df_1d['close'].iloc[prev_idx]
                    
                    # Calculate Camarilla levels
                    range_val = prev_high - prev_low
                    if range_val > 0:
                        camarilla_r1 = prev_close + (range_val * 1.1 / 12)
                        camarilla_s1 = prev_close - (range_val * 1.1 / 12)
                        
                        if position == 0:
                            # Long: price breaks above Camarilla R1 + volume spike > 2.0 + price above 1d EMA34
                            if price_close > camarilla_r1 and vol_spike > 2.0 and price_close > trend_1d:
                                signals[i] = 0.25
                                position = 1
                            # Short: price breaks below Camarilla S1 + volume spike > 2.0 + price below 1d EMA34
                            elif price_close < camarilla_s1 and vol_spike > 2.0 and price_close < trend_1d:
                                signals[i] = -0.25
                                position = -1
                        
                        elif position != 0:
                            # Simple ATR-based stoploss
                            if position == 1:  # Long
                                # Stop if price closes below entry - 2.0 * ATR
                                if price_close < (entry_price := getattr(generate_signals, 'entry_price_long', price_close)) - 2.0 * atr_val:
                                    signals[i] = 0.0
                                    position = 0
                                else:
                                    signals[i] = 0.25
                                    # Track entry price for stoploss
                                    if not hasattr(generate_signals, 'entry_price_long'):
                                        generate_signals.entry_price_long = price_close
                            else:  # Short
                                # Stop if price closes above entry + 2.0 * ATR
                                if price_close > (entry_price := getattr(generate_signals, 'entry_price_short', price_close)) + 2.0 * atr_val:
                                    signals[i] = 0.0
                                    position = 0
                                else:
                                    signals[i] = -0.25
                                    if not hasattr(generate_signals, 'entry_price_short'):
                                        generate_signals.entry_price_short = price_close
                    else:
                        if position != 0:
                            signals[i] = 0.0
                            position = 0
                else:
                    if position != 0:
                        signals[i] = 0.0
                        position = 0
            else:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
        else:
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0