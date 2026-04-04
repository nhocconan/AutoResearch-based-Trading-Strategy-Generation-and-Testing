#!/usr/bin/env python3
"""
Experiment #2623: 4h Donchian(20) breakout + 12h/1d EMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: 4h Donchian breakouts with 12h/1d trend alignment and volume confirmation capture 
institutional participation while minimizing whipsaws. Using 4h as primary timeframe targets 
75-200 trades over 4 years (19-50/year) to balance opportunity with fee efficiency. 
Volume spike requirement ensures momentum validity. Dual timeframe trend alignment (12h/1d) 
provides stronger trend filter than single timeframe, reducing false breakouts in choppy markets.
ATR-based trailing stop manages risk during adverse moves. Designed to work in both bull 
(riding trends) and bear (capturing breakdowns) markets by being directionally flexible.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2623_4h_donchian20_12h_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for EMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50)
    ema_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_12h = np.where(close_12h > ema_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian(20) channels, ATR(14) for volatility ===
    # Donchian channels (20-period high/low)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # True Range and ATR(14) for volatility-based stops
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size - balances opportunity with drawdown control
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators (max of 20, 14, 50)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_12h_aligned[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(atr[i])):
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
                # Exit if price breaks below Donchian low (mean reversion in strong trends)
                elif price < lowest_20[i]:
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
                # Exit if price breaks above Donchian high (mean reversion in strong trends)
                elif price > highest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require both 12h and 1d trend alignment for stronger bias filter
        trend_bias_12h = trend_12h_aligned[i]
        trend_bias_1d = trend_1d_aligned[i]
        
        # Only trade when both timeframes agree on trend direction
        if trend_bias_12h == 0 or trend_bias_1d == 0 or trend_bias_12h != trend_bias_1d:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: require significant volume spike (> 1.8x average)
        # Calculate volume ratio using 20-period volume moving average
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            vol_ratio = volume[i] / vol_ma if vol_ma > 0 else 0
            volume_spike = vol_ratio > 1.8
        else:
            volume_spike = False
        
        if volume_spike:
            # Long entry: price breaks above Donchian high with uptrend on both 12h and 1d
            if trend_bias_12h > 0 and trend_bias_1d > 0 and price > highest_20[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with downtrend on both 12h and 1d
            elif trend_bias_12h < 0 and trend_bias_1d < 0 and price < lowest_20[i]:
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