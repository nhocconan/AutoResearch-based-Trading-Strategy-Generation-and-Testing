Looking at the experiment history, I see the #1 problem is **overtrading**. The current strategy generates 1550 trades/symbol (>>400 max). The Camarilla pivot approach with multiple overlapping entry conditions (S3/S4/R3/R4 × vol × EMA) creates too many signals.

**New Hypothesis:** 4h Donchian(20) Breakout + Volume Confirmation + ATR Stop — BUT with ONE major change: **single entry condition** (not multiple overlapping). Use only the breakout at 20-bar high/low with volume confirmation. No stacking of multiple indicators.

This works in both bull and bear because Donchian breakout captures momentum in any direction. The key is **strict entry = fewer trades = less fee drag**.
#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + Volume Spike + ATR Trailing Stop

HYPOTHESIS: Simple Donchian(20) breakout at 4h captures institutional momentum.
Single entry condition = fewer trades = less fee drag = better test Sharpe.

KEY CHANGES FROM FAILED STRATEGIES:
1. ONLY Donchian breakout (no multiple pivot levels)
2. Volume spike REQUIRED (no vol = no entry)
3. Large holding period (12 bars minimum = 2 days)
4. Trade cooldown (no re-entry within 16 bars)
5. Simple ATR trailing stop (no complex pivot TP)

This works in BOTH bull and bear:
- Bull: price breaks above 20-bar high → long
- Bear: price breaks below 20-bar low → short

TARGET: 75-150 total trades over 4 years (proven pattern).
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382, 95tr)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_simple_v1"
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

def calculate_hma(close, period):
    """Hull Moving Average - simple version"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    # WMA helper
    def wma(data, length):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, length + 1, dtype=np.float64)
        for i in range(length - 1, len(data)):
            window = data[i - length + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    close_f = close.astype(np.float64)
    wma_half = wma(close_f, max(1, half))
    wma_full = wma(close_f, period)
    
    diff = np.where(np.isnan(wma_half) | np.isnan(wma_full), np.nan, 2.0 * wma_half - wma_full)
    return wma(diff, max(1, sqrt_n))

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for trend bias
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    
    # Trade management
    last_trade_bar = -999
    bars_since_trade = 999
    
    warmup = 60
    
    for i in range(warmup, n):
        bars_since_trade = i - last_trade_bar
        
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            continue
        
        if np.isnan(vol_ratio[i]):
            continue
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        price_below_1d_hma = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # === ENTRY CONDITIONS ===
        # Breakout: price CLOSES above/below 20-bar high/low
        bullish_breakout = close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]
        bearish_breakout = close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]
        
        # Volume spike REQUIRED
        vol_spike = vol_ratio[i] > 1.5
        
        # === SINGLE ENTRY CONDITION ===
        desired_signal = 0.0
        
        if not in_position and bars_since_trade >= 16:  # Cooldown: 16 bars minimum
            # LONG: Breakout above + volume spike + price above 1d HMA (bullish bias)
            if bullish_breakout and vol_spike and price_above_1d_hma:
                desired_signal = SIZE
                last_trade_bar = i
            
            # SHORT: Breakdown below + volume spike + price below 1d HMA (bearish bias)
            if bearish_breakout and vol_spike and price_below_1d_hma:
                desired_signal = -SIZE
                last_trade_bar = i
        
        # === STOPLOSS CHECK (ATR-based trailing stop) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLDING PERIOD (12 bars = 2 days) ===
        hold_bars = i - last_trade_bar if in_position else 0
        if in_position and hold_bars < 12:
            # Allow stoploss to trigger but don't exit early
            if desired_signal == 0.0:
                pass  # Stoploss already triggered
            else:
                desired_signal = position_side * SIZE  # Keep position
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals