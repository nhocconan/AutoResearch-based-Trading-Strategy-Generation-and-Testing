#!/usr/bin/env python3
"""
Experiment #4034: 1h Donchian(20) breakout + 4h/1d trend + volume confirmation + session filter
HYPOTHESIS: 1h Donchian breakouts aligned with 4h EMA50 trend (bullish when close > EMA50, bearish when close < EMA50) 
and 1d EMA200 regime filter (bullish when close > EMA200, bearish when close < EMA200) with volume confirmation 
(>1.5x MA20) capture high-probability continuation moves. Session filter (08-20 UTC) reduces noise trades. 
ATR(14) trailing stop (2.0x) controls drawdown. Discrete sizing (0.20) limits fee churn. Target: 60-150 total trades 
over 4 years (15-37/year). Works in bull/bear: In bull markets, buy upper breakouts; in bear markets, sell lower breakouts. 
Volume filter avoids whipsaws in ranging markets. Session filter focuses on active liquidity periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4034_1h_donchian20_4h_ema50_1d_ema200_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h EMA(50) for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 50:
        close_4h = pd.Series(df_4h['close'].values)
        ema_4h = close_4h.ewm(span=50, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d EMA(200) for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 200:
        close_1d = pd.Series(df_1d['close'].values)
        ema_1d = close_1d.ewm(span=200, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 1h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20 + 10, 50 + 10, 200 + 10)  # DC lookback, vol MA, 4h EMA buffer, 1d EMA buffer
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Trend filter: price relative to 4h EMA50
            price_above_4h_ema = price > ema_4h_aligned[i]
            price_below_4h_ema = price < ema_4h_aligned[i]
            
            # Regime filter: price relative to 1d EMA200
            price_above_1d_ema = price > ema_1d_aligned[i]
            price_below_1d_ema = price < ema_1d_aligned[i]
            
            # Breakout logic: 
            # - Bullish regime (price > 1d EMA200 AND price > 4h EMA50): look for long on upper Donchian breakout
            # - Bearish regime (price < 1d EMA200 AND price < 4h EMA50): look for short on lower Donchian breakout
            bullish_regime = price_above_1d_ema and price_above_4h_ema
            bearish_regime = price_below_1d_ema and price_below_4h_ema
            
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # Long conditions: bullish regime + upper Donchian breakout
            long_entry = bullish_regime and breakout_up
            
            # Short conditions: bearish regime + lower Donchian breakout
            short_entry = bearish_regime and breakout_down
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
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