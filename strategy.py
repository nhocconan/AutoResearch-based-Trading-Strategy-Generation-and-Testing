#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_RegimeFilter_ADX
Hypothesis: Camarilla R3/S3 breakouts filtered by 1d EMA34 trend, volume spike (>1.8x 20MA), and ADX regime filter (ADX>25) to avoid chop. Uses ATR trailing stop (2.0x) and discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 20-40 trades/year. Works in bull/bear by following 1d trend and requiring volume/ADX confirmation to avoid false breakouts in low-volatility regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4h ATR(10) for stoploss calculation
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr.ewm(span=10, adjust=False, min_periods=10).mean()
    atr_4h_values = atr_4h.values
    
    # Volume spike filter: volume > 1.8 * 20-period MA on 4h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    # ADX(14) for regime filter on 4h
    plus_dm = pd.Series(high).diff()
    minus_dm = pd.Series(low).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di_14 = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / tr_14)
    minus_di_14 = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / tr_14)
    dx = (abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    adx_filter = adx_values > 25  # Only trade when ADX > 25 (trending regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA34 (34), ATR (10), volume MA (20), ADX (14)
    start_idx = max(34, 10, 20, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        trend_val = ema34_1d_aligned[i]
        atr_val = atr_4h_values[i]
        vol_spike = volume_spike[i]
        adx_ok = adx_filter[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(atr_val) or np.isnan(adx_ok)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 1d EMA34 = uptrend, price < 1d EMA34 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Calculate Camarilla levels for previous 4h bar
        if i >= 1:
            # Use previous bar's high, low, close for today's Camarilla levels
            ph = high[i-1]
            pl = low[i-1]
            pc = close[i-1]
            rng = ph - pl
            # Camarilla R3 and S3 levels
            r3 = pc + (rng * 1.1 / 4)
            s3 = pc - (rng * 1.1 / 4)
        else:
            r3 = high_val
            s3 = low_val
        
        # Camarilla breakout conditions
        long_breakout = close_val > r3
        short_breakout = close_val < s3
        
        # Entry conditions: Camarilla breakout in direction of 1d trend + volume spike + ADX filter
        long_entry = long_breakout and is_uptrend and vol_spike and adx_ok
        short_entry = short_breakout and is_downtrend and vol_spike and adx_ok
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss (tighter for fewer whipsaws)
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.0 * ATR
            stop_price = highest_since_long - 2.0 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.0 * ATR
            stop_price = lowest_since_short + 2.0 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_RegimeFilter_ADX"
timeframe = "4h"
leverage = 1.0