#!/usr/bin/env python3
"""
Experiment #007: 6h Ichimoku Cloud + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Ichimoku TK Cross with cloud filter on 6h captures momentum, while 1d weekly pivot provides institutional bias. Volume confirmation filters false signals. Works in bull (TK cross above cloud with pivot support) and bear (TK cross below cloud with pivot resistance). Target: 75-150 total trades over 4 years (19-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_007_6h_ichimoku_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1d data for Weekly Pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Weekly Pivot from prior week (using last 5 trading days)
    # We approximate weekly pivot using daily data: (Weekly High + Weekly Low + Weekly Close) / 3
    # But we need prior week's values, so we use rolling window with offset
    if len(df_1d) >= 5:
        # Get weekly high/low/close from prior week (shift by 5 days to avoid look-ahead)
        weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().shift(5)
        weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().shift(5)
        weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().shift(5)
        
        # Weekly Pivot = (weekly_high + weekly_low + weekly_close) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Support/Resistance levels
        weekly_r1 = 2 * weekly_pivot - weekly_low
        weekly_s1 = 2 * weekly_pivot - weekly_high
        weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
        weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
        
        # Align to 6h timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1.values)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1.values)
        weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2.values)
        weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2.values)
    else:
        # Fallback if insufficient data
        weekly_pivot_aligned = np.full(n, np.nan)
        weekly_r1_aligned = np.full(n, np.nan)
        weekly_s1_aligned = np.full(n, np.nan)
        weekly_r2_aligned = np.full(n, np.nan)
        weekly_s2_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Ichimoku Cloud ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    lowest_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan_sen = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max()
    lowest_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun_sen = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    lowest_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used in signals to avoid look-ahead
    
    # Current cloud boundaries (Senkou Span A/B shifted forward 26 periods)
    # For trading, we use the cloud that was plotted 26 periods ago
    senkou_span_a_lagged = senkou_span_a.shift(26)
    senkou_span_b_lagged = senkou_span_b.shift(26)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_lagged.values, senkou_span_b_lagged.values)
    cloud_bottom = np.minimum(senkou_span_a_lagged.values, senkou_span_b_lagged.values)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # sufficient for Ichimoku (52+26) + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(tenkan_sen.iloc[i]) or np.isnan(kijun_sen.iloc[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan = tenkan_sen.iloc[i]
        kijun = kijun_sen.iloc[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Ichimoku Signals ---
        # TK Cross: Tenkan-sen crossing Kijun-sen
        tk_cross_up = tenkan > kijun and tenkan_sen.iloc[i-1] <= kijun_sen.iloc[i-1]
        tk_cross_down = tenkan < kijun and tenkan_sen.iloc[i-1] >= kijun_sen.iloc[i-1]
        
        # Price relative to cloud
        price_above_cloud = price > cloud_top[i]
        price_below_cloud = price < cloud_bottom[i]
        
        # --- Weekly Pivot Bias ---
        # Bullish bias: price above weekly pivot and S1
        bullish_bias = price > weekly_pivot_aligned[i] and price > weekly_s1_aligned[i]
        # Bearish bias: price below weekly pivot and R1
        bearish_bias = price < weekly_pivot_aligned[i] and price < weekly_r1_aligned[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for stoploss
            if i >= 14:
                tr = np.zeros(i+1)
                for j in range(1, i+1):
                    tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                tr[0] = high[0] - low[0]
                atr_val = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            else:
                atr_val = 0.0
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry (wider for 6h)
                stop_level = entry_price - 2.5 * atr_val
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr_val
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 28 bars (~112h on 6h)
            if bars_since_entry > 28:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: TK Cross Up + Price Above Cloud + Bullish Bias
            if tk_cross_up and price_above_cloud and bullish_bias:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: TK Cross Down + Price Below Cloud + Bearish Bias
            elif tk_cross_down and price_below_cloud and bearish_bias:
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