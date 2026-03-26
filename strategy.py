#!/usr/bin/env python3
"""
Experiment #009: 4h Donchian(16) Breakout + 1d EMA50 + Vol Spike

HYPOTHESIS: Donchian breakout is the SIMPLEST proven price channel.
Combined with 1d trend filter (EMA50), volume confirmation, and strict cooldown,
this should generate 75-150 total trades over 4 years (~20-40/year).

WHY THIS WORKS IN BULL AND BEAR:
- Bull: price breaks above Donchian upper → long, ride the trend
- Bear: price breaks below Donchian lower → short, catch the fall
- Range: choppiness filter prevents whipsaws
- Single condition = few trades = low fee drag

KEY CHANGES FROM #015 (1550 trades → target 75-150):
1. ONE entry path (Donchian only, not Camarilla S3/S4/R3/R4)
2. ALL filters required simultaneously (vol + EMA + ATR momentum)
3. 20-bar cooldown prevents re-entry spam
4. Stricter choppiness < 50

DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (95 trades, Sharpe 1.382)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian16_ema50_vol_cooldown_v1"
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

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging (no trades), CHOP < 50 = trending (allow trades)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction (align to 4h)
    ema_1d_raw = calculate_ema(df_1d['close'].values, span=50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ATR momentum: current ATR vs 20 bars ago
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_momentum = atr_14 / np.where(atr_ma > 0, atr_ma, 1)
    
    # Choppiness index
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (current vs 20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Donchian channels (16 bars)
    donchian_period = 16
    donchian_upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    bars_since_entry = 0
    last_trade_bar = -100  # Cooldown tracker
    
    # Warmup
    warmup = 80
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            in_position = False
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            in_position = False
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            continue
        
        # === REGIME CHECK: trending only ===
        chop = chop_14[i]
        is_trending = chop < 50.0
        
        # === COOLDOWN: at least 20 bars since last trade ===
        cooldown_ok = (i - last_trade_bar) >= 20
        
        # === ATR MOMENTUM: ATR rising for longs, falling for shorts ===
        # This ensures we're entering during volatility expansion
        atr_ok = atr_momentum[i] > 0.95 if not np.isnan(atr_momentum[i]) else True
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_ema = close[i] > ema_1d_aligned[i] if not np.isnan(ema_1d_aligned[i]) else True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if is_trending and cooldown_ok and atr_ok:
            # LONG: Break above Donchian upper + bullish 1d trend + volume
            if (close[i] > donchian_upper[i] and 
                price_above_ema and 
                vol_spike):
                desired_signal = SIZE
                last_trade_bar = i
            
            # SHORT: Break below Donchian lower + bearish 1d trend + volume
            if (close[i] < donchian_lower[i] and 
                not price_above_ema and 
                vol_spike):
                desired_signal = -SIZE
                last_trade_bar = i
        
        # === STOPLOSS CHECK (2x ATR) ===
        if in_position and position_side > 0:
            stop_loss = entry_price - 2.0 * entry_atr
            if low[i] < stop_loss:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            stop_loss = entry_price + 2.0 * entry_atr
            if high[i] > stop_loss:
                desired_signal = 0.0
        
        # === TAKE PROFIT: opposite Donchian band ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP at Donchian lower (mean reversion after breakout)
            if low[i] <= donchian_lower[i]:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP at Donchian upper
            if high[i] >= donchian_upper[i]:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                bars_since_entry = 0
        else:
            if in_position:
                bars_since_entry += 1
            else:
                bars_since_entry = 0
        
        # Exit if stopped out
        if desired_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            entry_atr = 0.0
            bars_since_entry = 0
        
        signals[i] = desired_signal
    
    return signals