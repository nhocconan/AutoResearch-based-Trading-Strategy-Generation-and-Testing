#!/usr/bin/env python3
"""
Experiment #028: 4h Donchian Breakout + Volume Spike + ADX Regime + ATR Stop

HYPOTHESIS: Combining Donchian breakout for price structure, ADX regime filter
for trend strength, volume spike confirmation for institutional participation,
and ATR trailing stop for risk management. Uses 1d SMA50 for trend direction
to work in both bull and bear markets.

WHY 4h: Most proven timeframe from DB (41% keep rate), better trade frequency
than 6h/12h while avoiding overtrading of lower TFs.

WHY IT WORKS IN BOTH BULL AND BEAR:
- Long entries: price above 1d SMA50 (bull filter) + breakout + volume
- Short entries: price below 1d SMA50 (bear filter) + breakdown + volume
- ADX > 25 filters out choppy/range-bound markets
- Symmetrical Donchian channels work for both breakout and breakdown

KEY IMPROVEMENTS OVER FAILED STRATEGIES:
- STRICTER entry: require CLOSE above Donchian high (not just intrabar touch)
- Volume spike 2.0x (not 1.5x) to reduce false breakouts
- ADX regime filter to avoid 2022 crash whipsaws
- Minimum 3-bar hold to avoid fee churn

TARGET: 75-150 total trades over 4 years. Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_adx_atr_v1"
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
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=20):
    """Average Directional Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_ema = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_ema = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm_ema[i] / atr[i]
            minus_di[i] = 100 * minus_dm_ema[i] / atr[i]
    
    dx = np.zeros(n, dtype=np.float64)
    for i in range(n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction (bull/bear filter)
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_20 = calculate_adx(high, low, close, period=20)
    
    # Donchian channels (20 periods = 5 days)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume - 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    trailing_stop = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
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
        
        if np.isnan(adx_20[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA50) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        price_below_1d_sma = close[i] < sma_1d_aligned[i]
        
        # === REGIME CHECK (ADX) ===
        # ADX > 25 indicates trending market (not choppy)
        is_trending = adx_20[i] > 25.0
        
        # Skip entries in choppy markets (ADX < 20)
        if adx_20[i] < 20.0 and not in_position:
            signals[i] = 0.0
            continue
        
        # Volume confirmation (2.0x for stricter filtering)
        vol_spike = vol_ratio[i] > 2.0
        
        # Previous bar's values for breakout detection
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Close breaks above Donchian high ===
            # Requires: price above SMA50 (bull), trending, volume spike
            if close[i] > prev_donchian_high and price_above_1d_sma:
                if vol_spike and is_trending:
                    desired_signal = SIZE
            
            # === SHORT ENTRY: Close breaks below Donchian low ===
            # Requires: price below SMA50 (bear), trending, volume spike
            if close[i] < prev_donchian_low and price_below_1d_sma:
                if vol_spike and is_trending:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === MINIMUM HOLD (3 bars) to reduce fee churn ===
        bars_held = i - entry_bar
        if in_position and bars_held < 3:
            # Only allow exits via stoploss
            pass
        elif in_position and bars_held >= 3:
            # Take profit at 3R
            if position_side > 0:
                profit_target = entry_price + 3.0 * entry_atr
                if close[i] >= profit_target:
                    desired_signal = 0.0  # Take profit
            elif position_side < 0:
                profit_target = entry_price - 3.0 * entry_atr
                if close[i] <= profit_target:
                    desired_signal = 0.0  # Take profit
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                trailing_stop = entry_price - 2.5 * entry_atr if position_side > 0 else entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                trailing_stop = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals