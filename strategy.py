#!/usr/bin/env python3
"""
EXPERIMENT #035 - MTF KAMA_Donchian_RSI Mean Reversion (1h base + 4h trend)
==================================================================================================
Hypothesis: Combine 4h KAMA+HMA trend filter + 1h Donchian breakout + RSI pullback confirmation.
This differs from current best by:
- KAMA (adaptive) instead of pure HMA for trend
- Donchian channel for breakout levels (vs pure RSI extremes)
- 1h base timeframe (more trades than 4h, cleaner than 15m)
- ATR-based dynamic position sizing (volatility-adjusted)

Why this should work:
- 4h KAMA adapts to market volatility (better than static MA in chop)
- Donchian breakout confirms momentum direction
- RSI pullback ensures we're not chasing extremes
- 1h timeframe has shown good trade frequency in past experiments
- Dynamic sizing reduces position when volatility spikes (drawdown control)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_donchian_rsi_1h_4h_v2"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[1:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=21):
    """Hull Moving Average"""
    half = period // 2
    sqrt_p = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    hma = pd.Series(2 * wma1 - wma2).ewm(span=sqrt_p, adjust=False).mean().values
    return hma


def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    
    if n < period:
        return kama
    
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i-period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        er[i] = signal / noise if noise > 0 else 0
    
    sc_fast = 2.0 / (fast + 1)
    sc_slow = 2.0 / (slow + 1)
    
    kama[period-1] = close[period-1]
    for i in range(period, n):
        sc = (er[i] * (sc_fast - sc_slow) + sc_slow) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama


def calculate_rsi(close, period=14):
    """RSI calculation"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_donchian(high, low, period=20):
    """Donchian Channel (upper/lower)"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    n = len(close)
    
    # 1h indicators (base timeframe)
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, period=21)
    kama_1h = calculate_kama(close, period=10)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # 4h trend filter using mtf_data helper
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        hma_4h = calculate_hma(c_4h, period=21)
        kama_4h = calculate_kama(c_4h, period=10)
        
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    except Exception:
        hma_4h_aligned = np.zeros(n)
        kama_4h_aligned = np.zeros(n)
    
    signals = np.zeros(n)
    
    # Position sizing parameters
    BASE_SIZE = 0.30  # Base position size (30% of capital)
    MAX_SIZE = 0.40   # Absolute max (40%)
    MIN_SIZE = 0.15   # Minimum when vol is high
    ATR_TARGET = 0.025  # Target ATR as % of price (2.5%)
    
    # Entry thresholds
    RSI_LONG_MAX = 45   # Long when RSI < 45 (pullback)
    RSI_SHORT_MIN = 55  # Short when RSI > 55 (rally)
    RSI_EXIT_LONG = 60  # Exit long when RSI > 60
    RSI_EXIT_SHORT = 40 # Exit short when RSI < 40
    
    # Donchian breakout confirmation
    DONCHIAN_BREAKOUT_PCT = 0.005  # 0.5% beyond channel for confirmation
    
    first_valid = max(100, 30 + 14, 20)
    
    # Track state for stoploss/takeprofit
    in_position = np.zeros(n, dtype=bool)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    highest_since = np.zeros(n)
    lowest_since = np.zeros(n)
    
    for i in range(first_valid, n):
        if atr_1h[i] == 0 or np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # 4h trend filter
        trend_4h = 0
        if i < len(hma_4h_aligned) and i < len(kama_4h_aligned):
            hma_val = hma_4h_aligned[i]
            kama_val = kama_4h_aligned[i]
            if hma_val > 0 and kama_val > 0:
                if price > hma_val and price > kama_val:
                    trend_4h = 1  # Bullish
                elif price < hma_val and price < kama_val:
                    trend_4h = -1  # Bearish
        
        # Dynamic position sizing based on ATR
        atr_pct = atr_1h[i] / price if price > 0 else 0.01
        if atr_pct > 0:
            size_mult = ATR_TARGET / atr_pct
            size_mult = np.clip(size_mult, 0.5, 1.5)  # Limit sizing adjustment
        else:
            size_mult = 1.0
        
        target_size = BASE_SIZE * size_mult
        target_size = np.clip(target_size, MIN_SIZE, MAX_SIZE)
        
        # Stoploss and takeprofit management
        if in_position[i-1]:
            side = position_side[i-1]
            entry = entry_price[i-1]
            highest = highest_since[i-1]
            lowest = lowest_since[i-1]
            
            # Update high/low since entry
            if side > 0:
                highest = max(highest, price)
                lowest = min(lowest, price) if lowest > 0 else price
            else:
                highest = max(highest, price) if highest > 0 else price
                lowest = min(lowest, price)
            
            highest_since[i] = highest
            lowest_since[i] = lowest
            
            # Stoploss: 2.5 * ATR
            stop_dist = 2.5 * atr_1h[i]
            if side > 0:
                stop_price = entry - stop_dist
                if price < stop_price:
                    signals[i] = 0.0
                    in_position[i] = False
                    position_side[i] = 0
                    continue
                
                # Take profit at 2R, reduce to half
                tp_price = entry + 2 * stop_dist
                if price >= tp_price and signals[i-1] > 0:
                    signals[i] = target_size * 0.5
                    in_position[i] = True
                    position_side[i] = 1
                    continue
                
                # Trail stop at 1R after TP hit
                if signals[i-1] <= target_size * 0.6:
                    trail_stop = highest - stop_dist
                    if price < trail_stop:
                        signals[i] = 0.0
                        in_position[i] = False
                        position_side[i] = 0
                        continue
                
                # RSI exit for longs
                if rsi_1h[i] > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    in_position[i] = False
                    position_side[i] = 0
                    continue
            else:
                stop_price = entry + stop_dist
                if price > stop_price:
                    signals[i] = 0.0
                    in_position[i] = False
                    position_side[i] = 0
                    continue
                
                # Take profit at 2R, reduce to half
                tp_price = entry - 2 * stop_dist
                if price <= tp_price and signals[i-1] < 0:
                    signals[i] = -target_size * 0.5
                    in_position[i] = True
                    position_side[i] = -1
                    continue
                
                # Trail stop at 1R after TP hit
                if signals[i-1] >= -target_size * 0.6:
                    trail_stop = lowest + stop_dist
                    if price > trail_stop:
                        signals[i] = 0.0
                        in_position[i] = False
                        position_side[i] = 0
                        continue
                
                # RSI exit for shorts
                if rsi_1h[i] < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    in_position[i] = False
                    position_side[i] = 0
                    continue
            
            # Hold position
            signals[i] = signals[i-1]
            in_position[i] = True
            position_side[i] = side
            entry_price[i] = entry
            continue
        
        # Entry logic
        # Long: 4h bullish + RSI pullback + price near Donchian lower (support)
        if trend_4h == 1 and rsi_1h[i] < RSI_LONG_MAX:
            donch_support = donch_lower[i]
            if price <= donch_support * (1 + DONCHIAN_BREAKOUT_PCT):
                signals[i] = target_size
                in_position[i] = True
                position_side[i] = 1
                entry_price[i] = price
                highest_since[i] = price
                lowest_since[i] = price
                continue
        
        # Short: 4h bearish + RSI rally + price near Donchian upper (resistance)
        elif trend_4h == -1 and rsi_1h[i] > RSI_SHORT_MIN:
            donch_resist = donch_upper[i]
            if price >= donch_resist * (1 - DONCHIAN_BREAKOUT_PCT):
                signals[i] = -target_size
                in_position[i] = True
                position_side[i] = -1
                entry_price[i] = price
                highest_since[i] = price
                lowest_since[i] = price
                continue
        
        signals[i] = 0.0
    
    return signals