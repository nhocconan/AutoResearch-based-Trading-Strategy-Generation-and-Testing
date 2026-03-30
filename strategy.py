#!/usr/bin/env python3
"""
Experiment #022: Donchian + Volume Spike + Choppiness (4h)

HYPOTHESIS: Simplified version of proven DB winner mtf_4h_chop_donchian_vol_regime_12h_v1
(SOLUSDT test Sharpe 1.491, 107 trades).

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Donchian breakout captures trend starts regardless of direction
- Choppiness Index filters ranging markets (best performer for bear/range)
- Volume spike confirms institutional conviction
- HTF 12h trend provides direction bias

KEY INSIGHT from DB: Most failures were from TOO MANY stacked conditions.
Winner strategies use 2-3 conditions max. This is a simplified Donchian approach.

TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_simple_v1"
timeframe = "4h"
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


def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower


def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market "choppiness"
    CHOP > 61.8 = ranging (use mean reversion)
    CHOP < 38.2 = trending (use trend following)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n, dtype=np.float64)
    
    for i in range(period - 1, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            # Choppiness formula: 100 * log10(sum ATR / range) / log10(period)
            chop[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop


def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # === HTF 12h trend (SMA 50 on close) ===
    htf_close = df_12h['close'].values
    htf_sma50 = pd.Series(htf_close).rolling(window=50, min_periods=50).mean().values
    htf_trend_up = htf_close > htf_sma50
    htf_trend_down = htf_close < htf_sma50
    
    # Align HTF to LTF (with shift for no look-ahead)
    htf_up_aligned = align_htf_to_ltf(prices, df_12h, htf_trend_up.astype(float))
    htf_down_aligned = align_htf_to_ltf(prices, df_12h, htf_trend_down.astype(float))
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20 (proven period from DB winners)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # Position size 28%
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    # Warmup: Donchian needs 20, chop needs 14, volume needs 20
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === CONDITION 1: Choppiness (trend filter) ===
        # Want trending market (CHOP < 61.8) but not too choppy
        is_trending = chop[i] < 61.8
        is_choppy = chop[i] > 61.8
        
        # === CONDITION 2: Volume spike ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === CONDITION 3: HTF trend alignment ===
        htf_bull = htf_up_aligned[i] > 0.5 if not np.isnan(htf_up_aligned[i]) else True
        htf_bear = htf_down_aligned[i] > 0.5 if not np.isnan(htf_down_aligned[i]) else True
        
        # === Donchian breakout signals ===
        # Long: price breaks above 20-bar high with volume
        bull_breakout = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]
        # Short: price breaks below 20-bar low with volume
        bear_breakout = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]
        
        # === ENTRY LOGIC (simplified - 2-3 conditions only) ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Donchian breakout + volume + trending market + HTF bull
            if bull_breakout and vol_spike and is_trending and htf_bull:
                desired_signal = SIZE
            
            # SHORT: Donchian breakdown + volume + trending market + HTF bear
            if bear_breakout and vol_spike and is_trending and htf_bear:
                desired_signal = -SIZE
        
        # === STOPLOSS and EXIT ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: trail stops only (2.5 ATR max loss)
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if market turns choppy
                if is_choppy:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: trail stops only (2.5 ATR max loss)
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if market turns choppy
                if is_choppy:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals