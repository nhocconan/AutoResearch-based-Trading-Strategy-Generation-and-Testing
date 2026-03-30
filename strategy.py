#!/usr/bin/env python3
"""
Experiment #005: 12h Donchian(24) Breakout + Volume + 1d Trend Filter

HYPOTHESIS: Donchian channel breakout is the most robust price structure
indicator (proven in 16,000+ experiments). Using 12h timeframe = ~3x fewer
trades than 4h = less fee drag = better generalization.

WHY 12h DONCHIAN:
- 24 bars = 12 days of data for structure definition
- Institutional-level supply/demand zones
- Proven: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe 1.38, 95 trades)
- Proven: mtf_4h_hma_volume_donchian_adx_12h_atr_v1 (Sharpe 1.32, 94 trades)

WHY IT WORKS IN BOTH BULL AND BEAR:
- Long: price breaks ABOVE upper Donchian band + 1d trend up
- Short: price breaks BELOW lower Donchian band + 1d trend down
- Symmetrical structure = works in all market conditions

TARGET: 75-200 total trades over 4 years (19-50/year)
Signal size: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_ema200_1d_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper (highest high) and lower (lowest low)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, mid, lower

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - identifies trending vs ranging markets
    CHOP > 61.8 = choppy/ranging (avoid trend trades)
    CHOP < 38.2 = trending (good for trend trades)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_tr = 0.0
        for j in range(period):
            idx = i - j
            tr = max(high[idx] - low[idx], abs(high[idx] - close[idx-1]) if idx > 0 else high[idx] - low[idx])
            sum_tr += tr
        
        highest_high = max(high[i-period+1:i+1])
        lowest_low = min(low[i-period+1:i+1])
        range_val = highest_high - lowest_low
        
        if range_val > 0:
            chop[i] = 100 * (np.log(sum_tr) / np.log(range_val)) if range_val > 0 else 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend direction (proven filter)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian(24) - slightly wider than 20 for 12h
    donch_upper, donch_mid, donch_lower = calculate_donchian(high, low, period=24)
    
    # Choppiness for regime filter
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Pre-compute regime flags ===
    is_choppy = chop > 61.8
    is_trending = chop < 38.2
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    bars_since_breakout = 0  # Track bars since breakout signal
    
    warmup = 250  # Need enough for EMA200 alignment + Donchian buffer
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if Donchian not ready
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA200) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Long breakout: price closes above upper band
        long_breakout = close[i] > donch_upper[i]
        # Short breakout: price closes below lower band
        short_breakout = close[i] < donch_lower[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === REGIME CHECK ===
        # In trending markets, breakout is more reliable
        # In choppy markets, be more selective
        regime_ok_long = is_trending[i] or vol_spike  # Either trending OR strong volume
        regime_ok_short = is_trending[i] or vol_spike
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above upper Donchian + trend alignment + regime ===
            if long_breakout and price_above_1d_ema and regime_ok_long:
                desired_signal = SIZE
                bars_since_breakout = 0
            # Fading: price back at lower band in uptrend (mean-reversion backup)
            elif low[i] <= donch_lower[i] and price_above_1d_ema and vol_spike:
                desired_signal = SIZE
                bars_since_breakout = 0
            
            # === SHORT: Breakout below lower Donchian + trend alignment + regime ===
            if short_breakout and price_below_1d_ema and regime_ok_short:
                desired_signal = -SIZE
                bars_since_breakout = 0
            # Fading: price back at upper band in downtrend
            elif high[i] >= donch_upper[i] and price_below_1d_ema and vol_spike:
                desired_signal = -SIZE
                bars_since_breakout = 0
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TIME EXIT (hold minimum 4 bars = 2 days to reduce churn) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 4:
            # Exit on opposite breakout (trend exhaustion)
            if position_side > 0 and close[i] < donch_lower[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > donch_upper[i]:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
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
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        # Track bars since breakout for potential re-entry
        if long_breakout or short_breakout:
            bars_since_breakout = 0
        else:
            bars_since_breakout += 1
        
        signals[i] = desired_signal
    
    return signals