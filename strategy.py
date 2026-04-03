#!/usr/bin/env python3
"""
Experiment #257: 4h Donchian Breakout + 1d/1w Trend Filter + Volume Spike
HYPOTHESIS: 4h Donchian(20) breakouts capture medium-term trends. 
1d EMA50 > EMA200 defines bull regime (long bias), EMA50 < EMA200 defines bear regime (short bias).
1w EMA50 > EMA200 confirms primary trend alignment (avoid counter-trend trades).
Volume spike (>2x 20-bar MA) confirms breakout strength. 
ATR(14) stoploss (2.5x) manages risk. Discrete sizing 0.25 balances return and fees.
Works in bull markets via long breakouts and bear markets via short breakdowns.
Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_257_4h_donchian_1d_1w_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA50/EMA200 regime filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema50_gt_ema200_1d = align_htf_to_ltf(prices, df_1d, ema_50_1d > ema_200_1d)
    
    # === HTF: 1w data for primary trend confirmation ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema50_gt_ema200_1w = align_htf_to_ltf(prices, df_1w, ema_50_1w > ema_200_1w)
    
    # === 4h Indicators: Donchian Channels (20) ===
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.max(high[:i+1])
            donchian_low[i] = np.min(low[:i+1])
        else:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # === 4h Indicators: ATR(14) for stoploss and volatility filter ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 200  # for 1d EMA200
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(ema50_gt_ema200_1d[i]) or np.isnan(ema50_gt_ema200_1w[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require strong volume spike (> 2x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Regime Filters ---
        # 1d: EMA50 > EMA200 = bull regime (long bias), EMA50 < EMA200 = bear regime (short bias)
        bull_regime_1d = ema50_gt_ema200_1d[i]
        bear_regime_1d = not bull_regime_1d
        
        # 1w: EMA50 > EMA200 = primary trend up (allow longs), EMA50 < EMA200 = primary trend down (allow shorts)
        bull_trend_1w = ema50_gt_ema200_1w[i]
        bear_trend_1w = not bull_trend_1w
        
        # --- Donchian Breakout Signals ---
        # Long breakout: price > Donchian high (20-bar)
        long_breakout = price > donchian_high[i]
        # Short breakout: price < Donchian low (20-bar)
        short_breakout = price < donchian_low[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite breakout (trailing)
                if short_breakout and volume_spike:
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
                # Exit on opposite breakout (trailing)
                if long_breakout and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if volume_spike:
            # Bull regime + bull trend: allow long breakouts
            if bull_regime_1d and bull_trend_1w:
                if long_breakout:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            # Bear regime + bear trend: allow short breakouts
            elif bear_regime_1d and bear_trend_1w:
                if short_breakout:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            # Counter-trend regime: require stronger volume confirmation
            else:
                # In mixed regimes, require very strong volume to trade counter-primary-trend
                if vol_ratio[i] > 3.0:  # Extreme volume spike
                    if long_breakout:
                        in_position = True
                        position_side = 1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = SIZE
                    elif short_breakout:
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