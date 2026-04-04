#!/usr/bin/env python3
"""
Experiment #3509: 4h Donchian Breakout + 1d/1w Trend + Volume Confirmation
HYPOTHESIS: 4h Donchian(20) breakouts with 1d EMA50 trend filter and 1w EMA200 regime filter capture medium-term momentum. 
1d EMA50 provides intermediate trend direction, 1w EMA200 defines bull/bear regime for bias adjustment. 
Volume confirms breakout strength. Position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
Uses 1d for trend filter and 1w for regime filter, 4h only for entry timing and risk management.
Works in bull (continuation from 1d trend support) and bear (continuation from 1d trend resistance via short) via price channels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3509_4h_donchian20_1d_1w_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === HTF: 1w data for regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA200 for regime filter (bull/bear)
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # === 4h Indicators: Donchian channels (20-period) for entry timing ===
    lookback_4h = 20
    highest_high_4h = pd.Series(high).rolling(window=lookback_4h, min_periods=lookback_4h).max().values
    lowest_low_4h = pd.Series(low).rolling(window=lookback_4h, min_periods=lookback_4h).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_4h, 50, 200, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_4h[i]) or np.isnan(lowest_low_4h[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below 1d EMA50 - trend reversal
                elif price < ema50_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above 1d EMA50 - trend reversal
                elif price > ema50_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) for confirmation
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine trend bias from 1d EMA50
            price_vs_ema50 = price - ema50_1d_aligned[i]
            
            # Regime filter: 1w EMA200 determines bull/bear market
            is_bull_regime = price > ema200_1w_aligned[i]
            
            # Long entry: price breaks above 4h Donchian high with bullish alignment
            if (price > highest_high_4h[i] and 
                price_vs_ema50 > 0 and  # Above 1d EMA50 = bullish bias
                (is_bull_regime or price_vs_ema50 > ema50_1d_aligned[i] * 0.02)):  # In bull regime OR strong bullish bias in bear regime
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 4h Donchian low with bearish alignment
            elif (price < lowest_low_4h[i] and 
                  price_vs_ema50 < 0 and  # Below 1d EMA50 = bearish bias
                  (not is_bull_regime or abs(price_vs_ema50) > ema50_1d_aligned[i] * 0.02)):  # In bear regime OR strong bearish bias in bull regime
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals