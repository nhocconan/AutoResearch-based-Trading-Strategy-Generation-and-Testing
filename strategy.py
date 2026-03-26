#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian Breakout + Volume + 1d Trend Filter

HYPOTHESIS: 12h Donchian(20) breakout captures medium-term institutional moves.
The 12h timeframe naturally reduces trade frequency vs 4h while maintaining
signal quality. Volume confirmation filters false breakouts, and 1d HMA
provides trend context to avoid trading against the major trend.

WHY 12h OVER 4h:
- Fewer trades (20-35/year vs 40-80/year) = less fee drag
- More reliable breakouts (longer consolidation = stronger signal)
- Institutional order flow more visible at this TF
- 54% keep rate proven in DB for 12h strategies

TARGET: 50-150 total trades over 4 years (~15-35/year)
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382, 95tr)

KEY ENTRY CONDITIONS (tight = fewer but better trades):
1. Donchian(20) breakout: price closes above/below 20-bar high/low
2. Volume confirmation: volume > 1.3x 20-bar MA
3. 1d HMA trend: aligned with entry direction
4. Stoploss: 2.5x ATR trailing

STOP TRADING RULES (prevents overtrading):
- Minimum 5 bars between trades (60h hold minimum)
- No re-entry after stop until opposite signal
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1d_hma_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators (vectorized for speed)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 bars = ~10 days on 12h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signal array
    signals = np.zeros(n)
    SIZE = 0.30  # Position size
    
    # === Position tracking state ===
    position_side = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    stop_price = 0.0
    
    # Minimum bars between trades (prevents overtrading)
    MIN_BARS_BETWEEN_TRADES = 5
    last_trade_bar = -MIN_BARS_BETWEEN_TRADES  # Allow first trade immediately
    
    warmup = max(50, donchian_period + 5)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            position_side = 0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        # === 1d HMA trend filter ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        trend_bullish = price_above_1d_hma
        trend_bearish = not price_above_1d_hma
        
        # === Volume confirmation ===
        vol_confirm = vol_ratio[i] > 1.3
        
        # === Donchian breakout detection ===
        prev_close = close[i - 1] if i > 0 else close[0]
        
        # Long signal: price breaks above 20-bar high
        bullish_breakout = close[i] > donchian_high[i] and prev_close <= donchian_high[i - 1] if i > 0 else False
        
        # Short signal: price breaks below 20-bar low
        bearish_breakout = close[i] < donchian_low[i] and prev_close >= donchian_low[i - 1] if i > 0 else False
        
        # === Bars since last trade ===
        bars_since_trade = i - last_trade_bar
        can_trade = bars_since_trade >= MIN_BARS_BETWEEN_TRADES
        
        # === Stoploss check (2.5 ATR) ===
        stoploss_triggered = False
        if position_side == 1:
            if low[i] < stop_price:
                stoploss_triggered = True
        elif position_side == -1:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        # === Exit on opposite signal or stoploss ===
        if stoploss_triggered or (position_side == 1 and bearish_breakout) or (position_side == -1 and bullish_breakout):
            signals[i] = 0.0
            if stoploss_triggered:
                last_trade_bar = i
            position_side = 0
            entry_price = 0.0
            entry_atr = 0.0
            entry_bar = 0
            stop_price = 0.0
            continue
        
        # === Entry logic ===
        desired_signal = 0.0
        
        if can_trade and position_side == 0:
            # LONG: Bullish breakout + volume confirm + bullish trend
            if bullish_breakout and vol_confirm and trend_bullish:
                desired_signal = SIZE
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                stop_price = entry_price - 2.5 * entry_atr
                last_trade_bar = i
            
            # SHORT: Bearish breakout + volume confirm + bearish trend
            elif bearish_breakout and vol_confirm and trend_bearish:
                desired_signal = -SIZE
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                stop_price = entry_price + 2.5 * entry_atr
                last_trade_bar = i
        
        # === Update trailing stop for existing positions ===
        elif position_side == 1:
            # Trail stop upward
            new_stop = close[i] - 2.5 * atr_14[i]
            stop_price = max(stop_price, new_stop)
            desired_signal = SIZE
        
        elif position_side == -1:
            # Trail stop downward
            new_stop = close[i] + 2.5 * atr_14[i]
            stop_price = min(stop_price, new_stop)
            desired_signal = -SIZE
        
        signals[i] = desired_signal
    
    return signals