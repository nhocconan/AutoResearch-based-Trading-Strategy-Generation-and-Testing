#!/usr/bin/env python3
"""
Experiment #2427: 6h Camarilla Pivot Breakout + 1d Volume Spike + Weekly Trend Filter
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
on 6h charts, combined with 1d volume spikes for confirmation and weekly trend filter, 
capture institutional order flow at key support/resistance levels. Works in bull markets 
(breakouts above R4 with volume) and bear markets (breakdowns below S4 with volume). 
Uses discrete position sizing (0.25) to limit fee drag and ensure 50-150 total trades 
over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2427_6h_camarilla_pivot_1d_vol_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and volume MA (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    rng = high_1d - low_1d
    camarilla_r4 = close_1d + 1.5 * rng
    camarilla_r3 = close_1d + 1.1 * rng
    camarilla_s3 = close_1d - 1.1 * rng
    camarilla_s4 = close_1d - 1.5 * rng
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed 1d bar only)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1d Volume MA(20) for spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones_like(volume_1d, dtype=np.float64)
    vol_ratio_1d[20:] = volume_1d[20:] / vol_ma_1d[20:]
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === HTF: 1w data for weekly trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_1w, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 6h Indicators: Price action for breakout/mean reversion ===
    # No additional 6h indicators needed beyond price itself
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # sufficient for weekly EMA and 1d indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(trend_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Trailing stop at 2*ATR using 6h Donchian width as proxy ---
        if in_position:
            # Calculate 6h Donchian width as ATR proxy (using 10-period)
            lookback_start = max(0, i-9)
            highest_10 = np.max(high[lookback_start:i+1])
            lowest_10 = np.min(low[lookback_start:i+1])
            donchian_width = highest_10 - lowest_10
            atr_estimate = donchian_width * 0.15  # approximate ATR from channel width
            
            if position_side > 0:  # Long position
                # Exit if price drops 2*ATR below entry OR breaks below S3 (mean reversion fail)
                if price < entry_price - 2.0 * atr_estimate or price < s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit if price rises 2*ATR above entry OR breaks above R3 (mean reversion fail)
                if price > entry_price + 2.0 * atr_estimate or price > r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Weekly trend filter: only trade in direction of weekly trend
        weekly_trend = trend_1w_aligned[i]
        
        # Volume confirmation: require 1d volume spike (> 2.0x average)
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        if not volume_spike:
            signals[i] = 0.0
            continue
        
        # Mean reversion at R3/S3 (fade extreme intraday moves)
        # Breakout continuation at R4/S4 (institutional participation)
        if weekly_trend > 0:  # Weekly uptrend bias
            # Long mean reversion: price rejects S3 with volume spike
            if price <= s3_aligned[i] and close[i-1] > s3_aligned[i-1]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Long breakout: price breaks above R4 with volume spike
            elif price >= r4_aligned[i] and close[i-1] < r4_aligned[i-1]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        else:  # Weekly downtrend bias
            # Short mean reversion: price rejects R3 with volume spike
            if price >= r3_aligned[i] and close[i-1] < r3_aligned[i-1]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            # Short breakdown: price breaks below S4 with volume spike
            elif price <= s4_aligned[i] and close[i-1] > s4_aligned[i-1]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals
</p>