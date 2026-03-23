#!/usr/bin/env python3
"""
Experiment #091: 4h Primary + 1d/1w HTF — Adaptive KAMA Trend with Choppiness Regime

Hypothesis: Previous 4h strategies failed due to overly strict regime filters blocking trades.
This version uses KAMA (Kaufman Adaptive Moving Average) which automatically adjusts to volatility,
combined with loose entry conditions to ensure 20-50 trades/year.

Key innovations:
1) KAMA adapts to market conditions — fast in trends, slow in chop (no separate regime switch needed)
2) 1d HMA for macro trend bias (proven in #079)
3) 1w HMA for ultra-long-term context (avoid counter-trend in major bear markets)
4) Choppiness Index modifies position size, doesn't block entries
5) RSI thresholds loosened (30-70 instead of 20-80) to generate more trades
6) ATR(14) trailing stoploss at 2.5x with signal→0 on breach

Why this should beat Sharpe=0.486:
- KAMA reduces whipsaw in choppy 2022-2023 period
- 1w HMA prevents major counter-trend trades in 2025 bear market
- Loose RSI ensures trades on ALL symbols (BTC/ETH/SOL)
- Position sizing scales with confluence (0.20 base, 0.35 max)

Position size: 0.20 base, 0.35 max with confluence
Stoploss: 2.5*ATR trailing
Target: 25-45 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_trend_1d1w_hma_chop_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise — fast in trends, slow in chop.
    Efficiency Ratio (ER) = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = [ER * (fast_SC - slow_SC) + slow_SC]^2
    """
    close_s = pd.Series(close)
    
    # Change = absolute difference over period
    change = np.abs(close_s - close_s.shift(period))
    
    # Volatility = sum of absolute differences
    volatility = np.abs(close_s.diff()).rolling(window=period, min_periods=period).sum()
    
    # Efficiency Ratio
    er = change / (volatility + 1e-10)
    er = er.fillna(0.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for ultra-long-term trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    kama_10 = calculate_kama(close, period=10)
    kama_30 = calculate_kama(close, period=30)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.20
    POSITION_SIZE_MAX = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(kama_10[i]) or np.isnan(kama_30[i]):
            continue
        if np.isnan(chop_14[i]):
            continue
        
        # === HTF TREND BIAS (1d and 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND SIGNAL (adaptive) ===
        kama_bullish = kama_10[i] > kama_30[i]
        kama_bearish = kama_10[i] < kama_30[i]
        
        # KAMA slope for confirmation
        kama_slope_long = kama_10[i] > kama_10[i-5] if i >= 5 else False
        kama_slope_short = kama_10[i] < kama_10[i-5] if i >= 5 else False
        
        # === CHOPPINESS REGIME (position size modifier, not hard filter) ===
        chop_trending = chop_14[i] < 50.0  # trending market
        chop_ranging = chop_14[i] > 50.0  # ranging market
        
        # === RSI ENTRY SIGNALS (loose thresholds for trade generation) ===
        rsi_not_overbought = rsi_14[i] < 65.0
        rsi_not_oversold = rsi_14[i] > 35.0
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        confluence_count = 0
        
        # --- LONG ENTRY ---
        long_conditions = 0
        
        # Primary: KAMA bullish crossover
        if kama_bullish:
            long_conditions += 1
        
        # 1d HMA bias (strong filter)
        if price_above_hma_1d:
            long_conditions += 1
        
        # 1w HMA ultra-long-term (soft filter - don't block, just reduce size if against)
        if price_above_hma_1w:
            long_conditions += 1
        
        # RSI not overbought
        if rsi_not_overbought:
            long_conditions += 1
        
        # KAMA slope confirmation
        if kama_slope_long:
            long_conditions += 1
        
        # Enter long if 3+ conditions met
        if long_conditions >= 3:
            new_signal = POSITION_SIZE_BASE
            confluence_count = long_conditions
            
            # Boost position size with confluence
            if confluence_count >= 4:
                new_signal = POSITION_SIZE_MAX
            if confluence_count >= 5:
                new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        short_conditions = 0
        
        # Primary: KAMA bearish crossover
        if kama_bearish:
            short_conditions += 1
        
        # 1d HMA bias (strong filter)
        if price_below_hma_1d:
            short_conditions += 1
        
        # 1w HMA ultra-long-term
        if price_below_hma_1w:
            short_conditions += 1
        
        # RSI not oversold
        if rsi_not_oversold:
            short_conditions += 1
        
        # KAMA slope confirmation
        if kama_slope_short:
            short_conditions += 1
        
        # Enter short if 3+ conditions met
        if short_conditions >= 3:
            new_signal = -POSITION_SIZE_BASE
            confluence_count = short_conditions
            
            # Boost position size with confluence
            if confluence_count >= 4:
                new_signal = -POSITION_SIZE_MAX
            if confluence_count >= 5:
                new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Keep position if RSI hasn't reached extreme exit zone
        if in_position and new_signal == 0.0:
            if position_side > 0 and rsi_14[i] < 75.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and rsi_14[i] > 25.0:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        # Exit long if 1d HMA turns bearish
        if in_position and position_side > 0:
            if price_below_hma_1d:
                new_signal = 0.0
        
        # Exit short if 1d HMA turns bullish
        if in_position and position_side < 0:
            if price_above_hma_1d:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals