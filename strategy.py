#!/usr/bin/env python3
"""
Experiment #028: 12h Donchian Breakout + 1d SMA200 Trend + Volume Confirmation

HYPOTHESIS: 12h Donchian(20) breakout captures institutional moves on medium-term
timeframe (10-day channel). 1d SMA200 provides HTF trend direction filter.
Volume confirmation ensures institutional participation. Choppiness avoids
range-bound whipsaws. 12h timeframe reduces trade frequency vs 4h/6h
while maintaining signal quality.

WHY 12h: Target 50-150 trades over 4 years. Previous 6h strategy overtraded (459 tr).
12h is 3x slower than 4h → should generate ~3x fewer trades = ~100-150 total.
54% keep rate observed in experiment #005.

WHY IT WORKS BOTH MARKETS:
- Symmetrical Donchian channels work for long breakouts (bull) AND short breakdowns (bear)
- 1d SMA200 filter adapts to bull/bear regime
- ATR stoploss handles volatility expansion in crashes

ENTRY CONDITIONS (tight = fewer trades = less fee drag):
1. Price breaks 20-bar high/low on 12h
2. Volume > 1.5x 20-bar average (institutional confirmation)
3. Price above 1d SMA200 for longs, below for shorts
4. Choppiness < 61.8 (avoiding range-bound markets)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1d_sma200_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - values above 61.8 = choppy/ranging, below = trending"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction (major trend filter)
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels (20 periods = 10 days on 12h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation (20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Conservative position sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = max(200, donchian_period + 20)  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if critical indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === REGIME FILTER (Choppiness Index) ===
        # Skip if too choppy (only enter when trending or neutral)
        is_choppy = chop[i] > 61.8
        
        # === DONCHIAN BREAKOUT VALUES (previous bar for signal) ===
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        # Volume confirmation (>1.5x average)
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Skip entries in choppy markets
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        if not in_position:
            # === LONG: Breakout above previous 20-bar high ===
            # Confirmed by: volume spike OR trending market (CHOP < 50)
            is_trending = chop[i] < 50.0
            breakout_long = high[i] > prev_donchian_high
            
            if breakout_long and price_above_1d_sma:
                if vol_spike or is_trending:
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below previous 20-bar low ===
            # Confirmed by: volume spike OR trending market
            breakout_short = low[i] < prev_donchian_low
            
            if breakout_short and not price_above_1d_sma:
                if vol_spike or is_trending:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR trailing stop) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT: Reduce at 3R profit ===
        if in_position and not stoploss_triggered:
            if position_side > 0:
                profit_r = (high[i] - entry_price) / entry_atr
                if profit_r >= 3.0:
                    desired_signal = SIZE / 2  # Take partial profit
            elif position_side < 0:
                profit_r = (entry_price - low[i]) / entry_atr
                if profit_r >= 3.0:
                    desired_signal = -SIZE / 2  # Take partial profit
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or direction flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals