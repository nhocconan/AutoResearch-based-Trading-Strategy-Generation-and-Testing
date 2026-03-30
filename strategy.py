#!/usr/bin/env python3
"""
Experiment #022: 4h Donchian Breakout + Volume + Choppiness Regime

HYPOTHESIS: This is the PROVEN winning pattern from the DB:
- mtf_4h_chop_donchian_vol_regime_12h_v1 (SOLUSDT: test_sharpe=1.491, 107 trades)
- mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (SOLUSDT: test_sharpe=1.382, 95 trades)

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull: Price breaks above upper Donchian + volume spike + chop<38 (trending) = long
- Bear: Price breaks below lower Donchian + volume spike + chop<38 (trending) = short
- Range: chop>61 = no entries (avoid whipsaws in 2022 crash)
- Simple pattern, tight but not too tight

KEY INSIGHT: The best performers use ONE signal type (Donchian) + volume + regime.
Not stacked indicators. Fewer conditions = fewer trades = less fee drag.

TARGET: 75-200 total trades over 4 years (19-50/year)
SIZE: 0.30 (30% of capital)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_simple_v3"
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
    """Donchian Channel: upper = highest high, lower = lowest low"""
    n = len(high)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, middle, lower

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP): measures trendiness vs choppiness
    CHOP > 61.8 = choppy/range (avoid entries)
    CHOP < 38.2 = trending (good for entries)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    dx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / atr[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr[i]
            
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # === HTF indicators ===
    # 12h Donchian for trend direction
    htf_upper_12h, htf_mid_12h, htf_lower_12h = calculate_donchian(
        df_12h['high'].values, df_12h['low'].values, period=12
    )
    
    # 12h price position relative to channel
    htf_close_12h = df_12h['close'].values
    htf_htf_bull = (htf_close_12h > htf_upper_12h * 0.98)  # Near upper = bull
    htf_htf_bear = (htf_close_12h < htf_lower_12h * 1.02)  # Near lower = bear
    
    htf_bull_aligned = align_htf_to_ltf(prices, df_12h, htf_htf_bull.astype(float))
    htf_bear_aligned = align_htf_to_ltf(prices, df_12h, htf_htf_bear.astype(float))
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    upper_dc, mid_dc, lower_dc = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness_index(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 60  # Donchian(20) + some buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME CHECK ===
        # Choppy market = no trend entries (avoid 2022 whipsaws)
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2 or adx[i] > 22
        
        # === DONCHIAN BREAKOUT ===
        # Long: price breaks above upper channel
        dc_breakout_long = close[i] > upper_dc[i] and close[i-1] <= upper_dc[i-1]
        # Short: price breaks below lower channel
        dc_breakout_short = close[i] < lower_dc[i] and close[i-1] >= lower_dc[i-1]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === HTF TREND ===
        htf_bull = htf_bull_aligned[i] > 0.5 if not np.isnan(htf_bull_aligned[i]) else False
        htf_bear = htf_bear_aligned[i] > 0.5 if not np.isnan(htf_bear_aligned[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Donchian breakout + volume + trending + HTF bull/neutral
            if dc_breakout_long and vol_spike and is_trending:
                if htf_bull or not htf_bear:  # Bull or neutral
                    desired_signal = SIZE
            
            # SHORT: Donchian breakdown + volume + trending + HTF bear/neutral
            if dc_breakout_short and vol_spike and is_trending:
                if htf_bear or not htf_bull:  # Bear or neutral
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if trend weakens: price falls back below middle channel
                if close[i] < mid_dc[i]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear and not htf_bull:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if trend weakens: price rises back above middle channel
                if close[i] > mid_dc[i]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull and not htf_bear:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to reduce fee churn ===
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