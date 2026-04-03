#!/usr/bin/env python3
"""
Experiment #074: 1h Donchian Breakout + Volume Spike + 4h/1d Regime Filter
HYPOTHESIS: 1h Donchian(20) breakouts with volume confirmation (>1.5x 20-bar average)
and aligned 4h/1d trend (price above/below EMA50) capture momentum in both bull/bear
markets. Using higher timeframes for direction reduces whipsaw, while 1h provides
timing. Session filter (08-20 UTC) avoids low-liquidity periods. Target: 60-150
trades over 4 years = 15-37/year on 1h timeframe. Position size fixed at 0.20
to control drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_074_1h_donchian_vol_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for EMA50 trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for EMA50 trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: Donchian Channels (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Session filter: 08-20 UTC (pre-compute for efficiency) ===
    # open_time is already datetime64[ms], use index for .hour
    hours = prices.index.hour  # DatetimeIndex.hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Fixed position size (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for Donchian and EMA stability
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if HTF data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 1.5  # Volume spike threshold
        
        # Determine trend from HTF EMAs
        uptrend_4h = price > ema_4h_aligned[i]
        uptrend_1d = price > ema_1d_aligned[i]
        downtrend_4h = price < ema_4h_aligned[i]
        downtrend_1d = price < ema_1d_aligned[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_long = False
            exit_short = False
            
            if position_side > 0:  # Long position
                # Exit on Donchian lower break or trend reversal
                if price < donchian_lower[i]:
                    exit_long = True
                elif not (uptrend_4h and uptrend_1d):  # Trend turned against us
                    exit_long = True
            else:  # Short position
                # Exit on Donchian upper break or trend reversal
                if price > donchian_upper[i]:
                    exit_short = True
                elif not (downtrend_4h and downtrend_1d):  # Trend turned against us
                    exit_short = True
            
            if exit_long or exit_short:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Minimum holding period of 3 bars
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long: price breaks above Donchian upper with volume and HTF uptrend
        if (price > donchian_upper[i-1] and vol_spike and 
            uptrend_4h and uptrend_1d):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        
        # Short: price breaks below Donchian lower with volume and HTF downtrend
        elif (price < donchian_lower[i-1] and vol_spike and 
              downtrend_4h and downtrend_1d):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals