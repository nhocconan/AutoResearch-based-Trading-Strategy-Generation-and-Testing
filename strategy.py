#!/usr/bin/env python3
"""
Experiment #024: 4h Donchian(20) Breakout + Volume + 12h Choppiness Regime

HYPOTHESIS: Donchian(20) breakouts are the most robust price structure signals.
Combined with volume confirmation and 12h choppiness regime filter:
- Bull market: Breakouts above 1d SMA = high probability continuation
- Bear market: Breakdown below 1d SMA = short setups work
- Range: Choppiness filter prevents whipsaws

KEY INSIGHT: Previous strategies either:
- Too loose (462 trades → overtrading)
- Too strict (0-4 trades → no edge)
- Need exactly: 4h TF + Donchian(20) + volume + regime

TRADE COUNT TARGET: 75-150 total over 4 years (19-37/year).
Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_vol_chop_12h_v2"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging (no trend) → don't trade breakout
    CHOP < 38.2 = trending → trade breakout
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = high[i-period:i+1].max()
        lowest = low[i-period:i+1].min()
        
        if highest - lowest > 1e-10:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian channel"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === 12h HTF data for regime filter (call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    chop_12h = calculate_choppiness(df_12h['high'].values, df_12h['low'].values, 
                                     df_12h['close'].values, period=14)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # 1d SMA for trend
    df_1d = get_htf_data(prices, '1d')
    sma_1d_50 = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    
    # === 4h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    dc_upper, dc_middle, dc_lower = calculate_donchian(high, low, period=20)
    
    # Volume analysis
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 80  # Need 20 bars for Donchian + 50 for SMA + buffer
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop_12h_aligned[i]) or np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === REGIME CHECK ===
        # Only trade when 12h chop indicates trending market
        chop_12h_trending = chop_12h_aligned[i] < 50.0  # Less choppy = trending
        
        # === TREND DIRECTION (1d SMA) ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above upper band
        breakout_up = close[i] > dc_upper[i] if not np.isnan(dc_upper[i]) else False
        # Breakdown below lower band
        breakout_down = close[i] < dc_lower[i] if not np.isnan(dc_lower[i]) else False
        
        # === MINIMUM HOLD: 2 bars (8h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR TRAILING STOP (2.5x ATR) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Trend reversal exits (only after min hold)
            if position_side > 0 and htf_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and htf_bullish and min_hold:
                stop_hit = True
            
            # Middle Donchian exit for profits
            if position_side > 0 and close[i] < dc_middle[i] and min_hold:
                stop_hit = True
            if position_side < 0 and close[i] > dc_middle[i] and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Breakout above Donchian upper + volume + 1d uptrend + trending
            if breakout_up and vol_spike and htf_bullish and chop_12h_trending:
                in_position = True
                position_side = 1
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # LONG SECONDARY: In uptrend, pullback to middle band with volume
            elif close[i] > sma_1d_aligned[i] and not np.isnan(dc_middle[i]):
                pullback_long = (close[i] < dc_middle[i] * 1.01) and vol_spike
                if pullback_long and chop_12h_trending:
                    in_position = True
                    position_side = 1
                    entry_atr = atr_14[i]
                    entry_bar = i
                    highest_since_entry = high[i]
                    signals[i] = SIZE * 0.5  # Half size for pullbacks
            
            # SHORT: Breakdown below Donchian lower + volume + 1d downtrend + trending
            elif breakout_down and vol_spike and htf_bearish and chop_12h_trending:
                in_position = True
                position_side = -1
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            # SHORT SECONDARY: In downtrend, rally to middle band with volume
            elif close[i] < sma_1d_aligned[i] and not np.isnan(dc_middle[i]):
                rally_short = (close[i] > dc_middle[i] * 0.99) and vol_spike
                if rally_short and chop_12h_trending:
                    in_position = True
                    position_side = -1
                    entry_atr = atr_14[i]
                    entry_bar = i
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE * 0.5  # Half size for rallies
            
            else:
                signals[i] = 0.0
    
    return signals