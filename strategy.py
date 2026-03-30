#!/usr/bin/env python3
"""
Experiment #023: Elder Ray + Donchian Breakout + Weekly Ichimoku (6h)

HYPOTHESIS: Combine structural price breaks (Donchian), momentum confirmation
(Elder Ray), and weekly trend filter (Ichimoku) for robust 6h entries.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price breaks above Donchian high + positive bull power + weekly bull = strong
- Bear: Price breaks below Donchian low + negative bear power + weekly bear = strong
- Weekly Ichimoku keeps us aligned with dominant trend, avoids whipsaws at major tops/bottoms

TIMEFRAME: 6h (1460 bars/year, 5840 over 4 years)
TARGET TRADES: 50-150 total (12-37/year) - tight but achievable with multi-filter

KEY COMPONENTS (4 filters = tight but not too tight):
1. Weekly Ichimoku Cloud (1w) - HTF trend direction
2. Donchian(20) breakout on 6h - structural price breaks
3. Elder Ray bull/bear power - momentum confirmation
4. Volume spike - trade validation

WHY NOVEL: Ichimoku mentioned as "untried" in DB, Elder Ray never mentioned.
Simple dual-indicator combination with clear rules.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_donchian_weekly_ichimoku_v1"
timeframe = "6h"
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


def calculate_ichimoku(high, low, close, period_fast=9, period_medium=26, period_slow=52):
    """
    Ichimoku Cloud calculation
    Returns: tenkan, kijun, senkou_a, senkou_b
    """
    n = len(close)
    
    # Tenkan-sen (Conversion Line): (9-period HH + 9-period LL) / 2
    tenkan = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period_fast + 1)
        hh = np.max(high[start:i+1])
        ll = np.min(low[start:i+1])
        tenkan[i] = (hh + ll) / 2
    
    # Kijun-sen (Base Line): (26-period HH + 26-period LL) / 2
    kijun = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period_medium + 1)
        hh = np.max(high[start:i+1])
        ll = np.min(low[start:i+1])
        kijun[i] = (hh + ll) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = np.full(n, np.nan, dtype=np.float64)
    for i in range(period_medium, n):
        senkou_a[i] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period HH + 52-period LL) / 2
    senkou_b = np.full(n, np.nan, dtype=np.float64)
    for i in range(period_slow, n):
        start = i - period_slow + 1
        hh = np.max(high[start:i+1])
        ll = np.min(low[start:i+1])
        senkou_b[i] = (hh + ll) / 2
    
    return tenkan, kijun, senkou_a, senkou_b


def calculate_donchian(high, low, period=20):
    """Donchian Channel - price channel breakout structure"""
    n = len(high)
    upper = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if i >= period - 1:
            start = i - period + 1
            upper[i] = np.max(high[start:i+1])
            lower[i] = np.min(low[start:i+1])
        else:
            upper[i] = high[i]
            lower[i] = low[i]
    
    return upper, lower


def calculate_elder_ray(high, low, close, ema_period=13):
    """
    Elder Ray - Bull Power and Bear Power
    Bull Power = High - EMA(close, 13) - measures buying pressure
    Bear Power = Low - EMA(close, 13) - measures selling pressure
    """
    ema_val = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    bull_power = high - ema_val
    bear_power = low - ema_val
    return bull_power, bear_power, ema_val


def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # ============================================================
    # STEP 1: Load HTF data ONCE before loop
    # ============================================================
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Ichimoku for HTF trend
    tenkan_w, kijun_w, senkou_a_w, senkou_b_w = calculate_ichimoku(
        df_1w['high'].values, df_1w['low'].values, df_1w['close'].values
    )
    
    # Weekly trend: price above BOTH cloud lines = bull, below BOTH = bear
    weekly_price = df_1w['close'].values
    weekly_bullish = (weekly_price > senkou_a_w) & (weekly_price > senkou_b_w)
    weekly_bearish = (weekly_price < senkou_a_w) & (weekly_price < senkou_b_w)
    
    # Align to 6h (shift by 1 for non-repainting)
    weekly_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bear_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # ============================================================
    # STEP 2: Pre-compute all indicators (vectorized where possible)
    # ============================================================
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    bull_power, bear_power, ema_13 = calculate_elder_ray(high, low, close, ema_period=13)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    # ============================================================
    # STEP 3: Signal generation with loop
    # ============================================================
    signals = np.zeros(n, dtype=np.float64)
    SIZE = 0.28  # Position size (within 0.25-0.30 range)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    # Warmup: Ichimoku needs 52, Elder Ray needs 13, volume needs 20
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            continue
        
        # ============================================================
        # FILTER 1: Weekly Ichimoku trend (HTF)
        # ============================================================
        htf_bull = weekly_bull_aligned[i] > 0.5 if not np.isnan(weekly_bull_aligned[i]) else False
        htf_bear = weekly_bear_aligned[i] > 0.5 if not np.isnan(weekly_bear_aligned[i]) else False
        
        # ============================================================
        # FILTER 2: Donchian breakout (6h structural)
        # ============================================================
        price_above_donchian = close[i] > donchian_upper[i]
        price_below_donchian = close[i] < donchian_lower[i]
        
        # Breakout = close above/below AND prior bar was not
        bull_breakout = price_above_donchian and not (close[i-1] > donchian_upper[i-1]) if i > 0 else price_above_donchian
        bear_breakout = price_below_donchian and not (close[i-1] < donchian_lower[i-1]) if i > 0 else price_below_donchian
        
        # ============================================================
        # FILTER 3: Elder Ray momentum (bull power positive = buying pressure)
        # ============================================================
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        # ============================================================
        # FILTER 4: Volume confirmation
        # ============================================================
        vol_spike = vol_ratio[i] > 1.5
        
        # ============================================================
        # ENTRY LOGIC
        # ============================================================
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Weekly bull + Donchian breakout + Bull power positive + Volume
            # All 4 filters must agree
            long_conditions = (htf_bull or not htf_bear) and bull_breakout and bull_power_positive and vol_spike
            
            if long_conditions:
                desired_signal = SIZE
            
            # SHORT: Weekly bear + Donchian breakout + Bear power negative + Volume
            short_conditions = (htf_bear or not htf_bull) and bear_breakout and bear_power_negative and vol_spike
            
            if short_conditions:
                desired_signal = -SIZE
        
        # ============================================================
        # EXIT LOGIC: ATR trailing stop
        # ============================================================
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # ATR-based trailing stop (2.5 ATR)
                stop_price = trailing_high - 2.5 * entry_atr
                
                # Stop out if price falls below stop
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if Elder Ray turns negative (momentum shift)
                if bear_power[i] < -0.5 * atr_14[i]:
                    desired_signal = 0.0
                
                # Exit if weekly turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # ATR-based trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                
                # Stop out if price rises above stop
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if Elder Ray turns positive (momentum shift)
                if bull_power[i] > 0.5 * atr_14[i]:
                    desired_signal = 0.0
                
                # Exit if weekly turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # ============================================================
        # MINIMUM HOLD: 6 bars (1.5 days) to avoid fee churn
        # ============================================================
        if in_position and (i - entry_bar) < 6:
            desired_signal = position_side * SIZE
        
        # ============================================================
        # UPDATE POSITION
        # ============================================================
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or reversal
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
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