#!/usr/bin/env python3
"""
Experiment #028: 12h Donchian(20) breakout + 1w/1d trend filter + volume confirmation
HYPOTHESIS: Donchian(20) breakouts on 12h aligned with weekly trend (price above/below weekly EMA50) 
and daily momentum (RSI(14) > 50 for longs, < 50 for shorts) capture institutional moves. 
Weekly trend filter ensures we trade with the dominant higher-timeframe momentum, 
avoiding counter-trend whipsaws in both bull and bear markets. Daily RSI acts as a 
momentum confirmation filter. Volume spike (>2.0x MA20) ensures participation. 
ATR stoploss (2.5x) and minimum 3-bar holding period reduce churn. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_028_12h_donchian20_1w_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 for trend filter
    weekly_ema50 = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # === HTF: 1d data for daily momentum (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Daily RSI(14) for momentum confirmation
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = rsi_14.fillna(50).values  # Neutral RSI for warmup
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # === 12h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr_12h = np.zeros(n)
    tr_12h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_12h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = 60  # Warmup for weekly EMA and daily RSI stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(weekly_ema50_aligned[i]) or np.isnan(rsi_14_aligned[i]) or np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Trend Filter: Price above/below weekly EMA50 ---
        price = close[i]
        is_uptrend = price > weekly_ema50_aligned[i]
        is_downtrend = price < weekly_ema50_aligned[i]
        
        # --- Daily Momentum Filter: RSI confirmation ---
        rsi = rsi_14_aligned[i]
        rsi_long = rsi > 50  # Bullish momentum
        rsi_short = rsi < 50  # Bearish momentum
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]  # Break above upper channel
        breakout_down = low[i] < donch_lower[i-1]  # Break below lower channel
        
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
                # Exit on opposite Donchian breakout (contrarian exit)
                if breakout_down and volume_spike:
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
                # Exit on opposite Donchian breakout (contrarian exit)
                if breakout_up and volume_spike:
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
        # Long: Donchian breakout up AND weekly uptrend AND daily bullish momentum AND volume spike
        if breakout_up and is_uptrend and rsi_long and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Donchian breakout down AND weekly downtrend AND daily bearish momentum AND volume spike
        elif breakout_down and is_downtrend and rsi_short and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals