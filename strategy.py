#!/usr/bin/env python3
"""
Experiment #087: 6h Ichimoku Cloud + Daily Trend Filter + Volume Spike

HYPOTHESIS: Ichimoku cloud (Tenkan/Kijun cross + price vs cloud) provides high-probability trend signals. 
Filtering by daily trend (price > daily EMA50 = bullish, < daily EMA50 = bearish) aligns with higher timeframe momentum. 
Volume spikes (>1.8x average) confirm breakout strength. Ichimoku works in both bull (cloud acts as support/resistance) 
and bear (cloud acts as resistance/support) markets. 6h timeframe targets 12-37 trades/year to minimize fee drag while 
capturing sustained trends. ATR-based stoploss manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_087_6h_ichimoku_daily_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for daily EMA50 trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 from 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        # Align to LTF (6h) timeframe with shift(1) for completed bars only
        daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        daily_ema50_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Ichimoku Components ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan = np.full(n, np.nan)
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun = np.full(n, np.nan)
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = np.full(n, np.nan)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_b = np.full(n, np.nan)
    
    # Calculate Tenkan and Kijun
    for i in range(n):
        if i >= 8:  # 9 periods need 8 previous + current
            tenkan[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
        if i >= 25:  # 26 periods need 25 previous + current
            kijun[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    # Calculate Senkou Span A and B
    for i in range(n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2
        if i >= 51:  # 52 periods need 51 previous + current
            senkou_b[i] = (np.max(high[i-51:i+1]) + np.min(low[i-51:i+1])) / 2
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Ensure enough data for Ichimoku (52 periods) and HTF daily EMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(daily_ema50_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Daily Trend Filter: Price > daily EMA50 = bullish bias, Price < daily EMA50 = bearish bias ---
        price_above_daily_ema50 = close[i] > daily_ema50_aligned[i]
        price_below_daily_ema50 = close[i] < daily_ema50_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Ichimoku Entry Conditions ---
        # Bullish: Tenkan > Kijun AND price > Senkou Span A AND price > Senkou Span B (price above cloud)
        ichimoku_bullish = (tenkan[i] > kijun[i]) and (close[i] > senkou_a[i]) and (close[i] > senkou_b[i])
        
        # Bearish: Tenkan < Kijun AND price < Senkou Span A AND price < Senkou Span B (price below cloud)
        ichimoku_bearish = (tenkan[i] < kijun[i]) and (close[i] < senkou_a[i]) and (close[i] < senkou_b[i])
        
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
                # Exit on Tenkan/Kijun cross reversal
                if tenkan[i] < kijun[i]:
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
                # Exit on Tenkan/Kijun cross reversal
                if tenkan[i] > kijun[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Ichimoku bullish + volume spike + price above daily EMA50
        long_condition = ichimoku_bullish and volume_spike and price_above_daily_ema50
        
        # Short: Ichimoku bearish + volume spike + price below daily EMA50
        short_condition = ichimoku_bearish and volume_spike and price_below_daily_ema50
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals