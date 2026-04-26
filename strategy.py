#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_ChopRegime
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts aligned with 1d EMA34 trend, volume spike (>2.0x 20-period MA), and choppy market regime (Choppiness Index > 61.8) capture high-probability mean-reversion bounces in both bull and bear markets. Uses discrete position sizing (0.0, ±0.25) and 4h ATR-based trailing stop (2.5x) for exits. Targets 20-40 trades/year by requiring confluence of HTF trend, volume confirmation, and regime filter—designed to work in ranging markets where breakouts fail and mean reversion prevails.
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
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d ATR(14) for trailing stop
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean()
    atr_1d_values = atr_1d.values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_values)
    
    # Volume spike filter: volume > 2.0 * 20-period MA on 4h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    # Choppiness Index regime filter: CHOP(14) > 61.8 = ranging market (mean revert)
    def calculate_chop(high, low, close, window=14):
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_sum = tr.rolling(window=window, min_periods=window).sum()
        hh = pd.Series(high).rolling(window=window, min_periods=window).max()
        ll = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(window)
        return chop.values
    
    chop_values = calculate_chop(high, low, close, 14)
    chop_regime = chop_values > 61.8  # ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA34 (34), ATR (14), volume MA (20), chop (14)
    start_idx = max(34, 14, 20, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        trend_val = ema34_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        vol_spike = volume_spike[i]
        regime = chop_regime[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(atr_val) or np.isnan(chop_values[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Only trade in ranging regime (choppy market)
        if not regime:
            # Hold current position or go flat
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
            lowest_since_short = 0.0
            continue
        
        # Trend filter: price > 1d EMA34 = uptrend bias, price < 1d EMA34 = downtrend bias
        # But in choppy regime, we mean revert: long when price < EMA, short when price > EMA
        is_above_ema = close_val > trend_val
        is_below_ema = close_val < trend_val
        
        # Calculate Camarilla levels for previous 4h bar
        if i >= 1:
            # Use previous bar's high, low, close for today's Camarilla levels
            ph = high[i-1]
            pl = low[i-1]
            pc = close[i-1]
            rng = ph - pl
            # Camarilla R1 and S1 levels (inner levels for mean reversion)
            r1 = pc + (rng * 1.1 / 12)  # R1 = C + (H-L)*1.1/12
            s1 = pc - (rng * 1.1 / 12)  # S1 = C - (H-L)*1.1/12
        else:
            r1 = high_val
            s1 = low_val
        
        # Camarilla mean reversion conditions
        long_setup = close_val < s1  # price below S1 = oversold
        short_setup = close_val > r1  # price above R1 = overbought
        
        # Entry conditions: Camarilla mean reversion opposite to 1d EMA bias + volume spike
        long_entry = long_setup and is_above_ema and vol_spike  # price < S1 but above EMA = bullish bounce
        short_entry = short_setup and is_below_ema and vol_spike  # price > R1 but below EMA = bearish bounce
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.5 * ATR
            stop_price = highest_since_long - 2.5 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.5 * ATR
            stop_price = lowest_since_short + 2.5 * atr_val
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

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0