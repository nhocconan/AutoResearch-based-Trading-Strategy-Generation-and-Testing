#!/usr/bin/env python3
"""
Experiment #5275: 6h Williams %R + 1d Volume Spike + 1w Trend Filter
HYPOTHESIS: On 6h timeframe, Williams %R identifies oversold/overbought conditions, filtered by 1d volume spikes (institutional interest) and 1w EMA50 trend direction. In bull 1w trend (price > EMA50), we go long when %R < -80 and volume > 1.5x 20-period average. In bear 1w trend (price < EMA50), we go short when %R > -20 and volume > 1.5x 20-period average. Uses discrete position sizing (0.25) to balance profit potential with drawdown control. Designed for 15-30 trades/year on 6h timeframe (60-120 total over 4 years) to minimize fee drag. Works in bull markets by buying the dip with volume confirmation and in bear markets by selling the rally with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5275_6h_williamsr_1d_volspike_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 1w data for trend filter (EMA50) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, min_periods=50, adjust=False).mean().shift(1).values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume spike detection ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().shift(1).values
        vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    else:
        vol_ma_20_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Williams %R (14-period) ===
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(14, 20, 50)  # Williams %R, volume MA, EMA50 warmup
    
    for i in range(warmup, n):
        # --- Session Filter: 00-24 UTC (6h timeframe, less restrictive) ---
        # 6h candles already filter to specific sessions, so we can use full day
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        # --- Exit Logic: Close position when Williams %R reverses or volume drops ---
        if in_position:
            # Check for Williams %R reversal (exit extreme zone)
            williams_r_extreme_long = williams_r[i] < -80  # Still oversold
            williams_r_extreme_short = williams_r[i] > -20  # Still overbought
            
            # Check volume condition (still elevated)
            volume_spike = vol_ratio > 1.5
            
            # Check trend consistency
            trend_bullish = price > ema_50_1w_aligned[i]
            trend_bearish = price < ema_50_1w_aligned[i]
            
            # Exit conditions:
            # 1. Williams %R exits extreme zone (mean reversion signal)
            # 2. Volume drops below spike level
            # 3. Trend changes (price crosses 1w EMA50)
            if position_side > 0:  # Long position
                if (williams_r[i] >= -80) or (not volume_spike) or (not trend_bullish):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if (williams_r[i] <= -20) or (not volume_spike) or (not trend_bearish):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Williams %R extreme levels
        williams_r_oversold = williams_r[i] < -80
        williams_r_overbought = williams_r[i] > -20
        
        # Volume spike confirmation (institutional interest)
        volume_spike = vol_ratio > 1.5
        
        # Trend filter from 1w
        trend_bullish = price > ema_50_1w_aligned[i]
        trend_bearish = price < ema_50_1w_aligned[i]
        
        # Entry conditions: Williams %R extreme + volume spike + trend alignment
        if williams_r_oversold and volume_spike and trend_bullish:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif williams_r_overbought and volume_spike and trend_bearish:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals