#!/usr/bin/env python3
"""
Experiment #189: 4h Primary + 1d/1w HTF — Fisher Transform + KAMA Trend + ADX Regime

Hypothesis: Previous CRSI+Choppiness strategies work but can be improved by:
1. Using Ehlers Fisher Transform for sharper reversal signals (better in bear markets)
2. KAMA adapts to volatility better than HMA/EMA (less whipsaw in 2022 crash)
3. ADX with hysteresis (enter>25, exit<20) for cleaner regime detection
4. Asymmetric entries: only short when 1d KAMA bearish, only long when 1d KAMA bullish
5. ATR-based position scaling: reduce size when vol spikes (protects in crashes)
6. Simpler hold logic to ensure positions persist (fixes 0-trade issue)

Key improvements over #184:
- Fisher Transform crosses are cleaner than CRSI extremes
- KAMA adapts ER (Efficiency Ratio) to market conditions
- ADX hysteresis prevents rapid regime flipping
- Volume confirmation on breakouts (reduces false signals)
- Better stoploss tracking (fixes bug in lowest_since_entry)
- Looser entry thresholds to ensure 30-50 trades/year

TARGET: 35-55 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_adx_regime_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market efficiency - moves fast in trends, slow in chop.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        change = np.abs(close[i] - close[i-er_period])
        vol = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        if vol > 1e-10:
            er[i] = change / vol
        else:
            er[i] = 0.0
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            trigger[i] = fisher[i]
            continue
        
        # Normalize price to -1 to +1
        x = (2.0 * (close[i] - lowest) / range_val) - 1.0
        x = np.clip(x, -0.999, 0.999)  # Prevent log errors
        
        # Fisher calculation
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        if i > 0:
            fisher[i] = 0.7 * fisher[i] + 0.3 * fisher[i-1]  # Smooth
        trigger[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, trigger

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.zeros(n)
    
    # True Range and DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI and DX
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
        minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # Calculate 1d KAMA for macro bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 1w KAMA for ultra-long-term trend
    kama_1w_raw = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # ADX hysteresis tracking
    prev_adx_regime = 0  # 0=neutral, 1=trend, 2=range
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] < 1e-10:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(kama_21[i]) or np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(adx_14[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === HTF MACRO BIAS ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        price_above_kama_1w = close[i] > kama_1w_aligned[i]
        price_below_kama_1w = close[i] < kama_1w_aligned[i]
        
        # === REGIME DETECTION (ADX with hysteresis) ===
        adx_trend = adx_14[i] > 25.0
        adx_range = adx_14[i] < 20.0
        
        # Hysteresis: only change regime if clear signal
        if adx_trend:
            adx_regime = 1  # trend
        elif adx_range:
            adx_regime = 2  # range
        else:
            adx_regime = prev_adx_regime  # hold previous
        prev_adx_regime = adx_regime
        
        # === VOLUME CONFIRMATION ===
        vol_above_avg = volume[i] > vol_sma[i] if not np.isnan(vol_sma[i]) else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Fisher Transform reversal signals
        fisher_long = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        fisher_short = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        
        # KAMA trend confirmation
        kama_bullish = close[i] > kama_21[i]
        kama_bearish = close[i] < kama_21[i]
        
        if adx_regime == 2:  # RANGE regime - mean reversion
            # Long: Fisher long signal + price above 1d KAMA (bullish bias)
            if fisher_long and price_above_kama_1d:
                if price_above_kama_1w:
                    new_signal = POSITION_SIZE_FULL
                else:
                    new_signal = POSITION_SIZE_HALF
            
            # Short: Fisher short signal + price below 1d KAMA (bearish bias)
            elif fisher_short and price_below_kama_1d:
                if price_below_kama_1w:
                    new_signal = -POSITION_SIZE_FULL
                else:
                    new_signal = -POSITION_SIZE_HALF
        
        elif adx_regime == 1:  # TREND regime - trend following
            # Long: KAMA bullish + price above 1d KAMA + volume confirmation
            if kama_bullish and price_above_kama_1d and vol_above_avg:
                if price_above_kama_1w:
                    new_signal = POSITION_SIZE_FULL
                else:
                    new_signal = POSITION_SIZE_HALF
            
            # Short: KAMA bearish + price below 1d KAMA + volume confirmation
            elif kama_bearish and price_below_kama_1d and vol_above_avg:
                if price_below_kama_1w:
                    new_signal = -POSITION_SIZE_FULL
                else:
                    new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # If already in position, hold unless exit conditions met
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 1d KAMA
                if price_above_kama_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 1d KAMA
                if price_below_kama_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i]) if lowest_since_entry > 0 else close[i]
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 1d KAMA (trend changed)
        if in_position and position_side > 0 and price_below_kama_1d:
            new_signal = 0.0
        
        # Exit short if price crosses above 1d KAMA (trend changed)
        if in_position and position_side < 0 and price_above_kama_1d:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_bar = i
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_bar = i
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_bar = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals