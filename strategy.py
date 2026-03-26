#!/usr/bin/env python3
"""
Experiment #021: 4h Keltner Squeeze + Donchian Direction + Choppiness Regime

HYPOTHESIS: Markets alternate between volatility contraction (squeeze) and 
expansion. When Bollinger Bands contract inside Keltner Channel (squeeze), 
volatility is compressed. A subsequent breakout through the contracted range 
often leads to large moves. Combined with Donchian for structure, Choppiness 
for regime filtering, and volume confirmation.

WHY 4h + 12h: 
- 4h captures multi-day swings with reasonable fee impact
- 12h confirms larger trend direction
- Proven timeframe from DB (Camarilla, TRIX winners on 4h)

KEY DIFFERENCE FROM DONCHIAN BREAKOUT: 
- Requires VOLATILITY CONTRACTION first (squeeze) before breakout signal
- This filters out breakouts in choppy markets
- Mean reversion after squeeze is also a valid signal

TARGET: 100-200 total trades over 4 years (25-50/year). HARD MAX: 300.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_keltner_squeeze_donchian_chop_v1"
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

def calculate_keltner(high, low, close, ema_period=20, atr_period=10, multiplier=2.0):
    """
    Keltner Channel
    Middle = EMA(20)
    Upper = EMA + multiplier * ATR(10)
    Lower = EMA - multiplier * ATR(10)
    """
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    
    return upper, middle, lower, ema, atr

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower

def calculate_donchian(high, low, period=20):
    """Donchian Channel - uses past period highs/lows"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, mid, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index
    CHOP < 38.2 = trending (momentum works)
    CHOP > 61.8 = choppy (mean reversion)
    """
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
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
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 for trend direction
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 12h ATR for stoploss
    atr_12h_raw = calculate_atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h_raw)
    
    # === Local 4h indicators ===
    # Keltner Channel
    kelt_upper, kelt_mid, kelt_lower, kelt_ema, kelt_atr = calculate_keltner(high, low, close, 
                                                                              ema_period=20, 
                                                                              atr_period=10, 
                                                                              multiplier=2.0)
    
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Squeeze detection: BB inside Keltner
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    kelt_width = (kelt_upper - kelt_lower) / (kelt_mid + 1e-10)
    
    # Squeeze when Bollinger bands are inside Keltner channels
    squeeze = (bb_upper < kelt_upper) & (bb_lower > kelt_lower)
    
    # Donchian for breakout structure
    donch_upper, donch_mid, donch_lower = calculate_donchian(high, low, period=20)
    
    # Choppiness for regime
    chop = calculate_choppiness(high, low, close, period=14)
    
    # ATR for stoploss
    atr_local = calculate_atr(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(kelt_atr[i]) or np.isnan(bb_width[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME FILTER ===
        is_trending = chop[i] < 38.2
        is_choppy = chop[i] > 61.8
        
        # === TREND DIRECTION (12h EMA) ===
        price_above_12h_ema = close[i] > ema_12h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === DONCHIAN BREAKOUT SIGNAL ===
        donch_breakout_up = close[i] > donch_upper[i]
        donch_breakout_down = close[i] < donch_lower[i]
        
        # === SQUEEZE STATE ===
        in_squeeze = squeeze[i]
        squeeze_prev = squeeze[i-1] if i > warmup else False
        squeeze_released = squeeze_prev and not in_squeeze  # Just released from squeeze
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === ENTRY: Squeeze release + Donchian breakout + volume ===
            # Trend following in trending markets
            if is_trending and price_above_12h_ema:
                if donch_breakout_up and vol_spike:
                    desired_signal = SIZE
            
            if is_trending and not price_above_12h_ema:
                if donch_breakout_down and vol_spike:
                    desired_signal = -SIZE
            
            # === ENTRY: Squeeze release reversal in choppy markets ===
            # Price mean-reverts after squeeze in range-bound markets
            if is_choppy:
                # Squeeze release is the signal for expansion
                if squeeze_released:
                    # Long: price near lower Keltner after squeeze release
                    if close[i] < kelt_lower[i] + 0.5 * kelt_atr[i]:
                        desired_signal = SIZE
                    
                    # Short: price near upper Keltner after squeeze release
                    if close[i] > kelt_upper[i] - 0.5 * kelt_atr[i]:
                        desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR) ===
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
        
        # === TIME-BASED EXIT (hold at least 6 bars = 1 day) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 6:
            # Exit if trend flips
            if position_side > 0 and not price_above_12h_ema:
                desired_signal = 0.0
            if position_side < 0 and price_above_12h_ema:
                desired_signal = 0.0
        
        # === RSI EXIT FILTER ===
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = (100 - (100 / (1 + rs)))[i]
        
        if in_position:
            if position_side > 0 and rsi > 80:
                desired_signal = 0.0
            if position_side < 0 and rsi < 20:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_local[i] if atr_local[i] > 0 else atr_12h_aligned[i]
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
        
        signals[i] = desired_signal
    
    return signals