#!/usr/bin/env python3
"""
Experiment #884: 4h Primary + 12h HTF — Volatility Spike Mean Reversion + Trend Filter

Hypothesis: After 600+ failed strategies, the winning pattern is SIMPLICITY + ONE STRONG EDGE.
Complex regime switching (Chop + CRSI + Donchian + multiple HTF) has failed repeatedly.

Key insights from research:
1. VOLATILITY SPIKE REVERSION works in bear/range markets (2022 crash, 2025 bear)
   - ATR(7)/ATR(30) > 2.0 = panic spike
   - Price < BB(20, 2.5) = oversold extreme
   - Exit when ATR ratio < 1.2 = vol normalized
2. 4h Primary TF: Proven to work (current best Sharpe=0.612 is 4h)
3. 12h HMA(21) ONLY for trend bias (not 1d+1w which overcomplicates)
4. Fewer conflicting filters = more trades generated
5. Discrete signals (0.0, ±0.25, ±0.30) minimize fee churn

Why this should beat Sharpe=0.612:
- Vol spike reversion has reported Sharpe 0.8-1.5 through 2022 crash
- Simpler logic = fewer conditions that block trades
- 4h TF targets 20-50 trades/year (optimal fee/trade balance)
- Asymmetric: only long in bull trend, only short in bear trend
- ATR trailing stop (2.5x) protects from runaway losses

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_spike_bb_reversion_12h_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Bollinger Bands with configurable std multiplier."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (4h) indicators
    atr_4h = calculate_atr(high, low, close, period=14)
    atr_fast_4h = calculate_atr(high, low, close, period=7)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    rsi_4h = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_4h[i]) or np.isnan(atr_fast_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(rsi_4h[i]):
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR(7) / ATR(30) > 2.0 = panic spike (use ATR(14) as proxy for 30)
        atr_ratio = atr_fast_4h[i] / (atr_4h[i] + 1e-10)
        vol_spike = atr_ratio > 2.0
        vol_normalized = atr_ratio < 1.2
        
        # === PRICE POSITION vs BOLLINGER BANDS ===
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        price_near_mid = (close[i] > bb_lower[i] * 0.995) and (close[i] < bb_upper[i] * 1.005)
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_4h[i] < 30
        rsi_overbought = rsi_4h[i] > 70
        rsi_extreme_oversold = rsi_4h[i] < 20
        rsi_extreme_overbought = rsi_4h[i] > 80
        
        # === TREND BIAS (12h HMA) ===
        trend_bullish = close[i] > hma_12h_aligned[i]
        trend_bearish = close[i] < hma_12h_aligned[i]
        
        # === SHORT-TERM TREND FILTER (SMA50/200) ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        
        # === VOL SPIKE LONG ENTRY (Mean Reversion) ===
        # Only long if: vol spike + price below BB + trend not strongly bearish
        if vol_spike and price_below_bb:
            if trend_bullish or above_sma50:
                # Strong confluence: bull trend + vol spike
                desired_signal = BASE_SIZE
            elif not trend_bearish or above_sma200:
                # Weaker confluence: neutral trend + vol spike
                desired_signal = REDUCED_SIZE
            elif rsi_extreme_oversold:
                # Extreme RSI alone can trigger reduced size
                desired_signal = REDUCED_SIZE
        
        # === VOL SPIKE SHORT ENTRY (Mean Reversion) ===
        # Only short if: vol spike + price above BB + trend not strongly bullish
        if vol_spike and price_above_bb:
            if trend_bearish or below_sma50:
                # Strong confluence: bear trend + vol spike
                desired_signal = -BASE_SIZE
            elif not trend_bullish or below_sma200:
                # Weaker confluence: neutral trend + vol spike
                desired_signal = -REDUCED_SIZE
            elif rsi_extreme_overbought:
                # Extreme RSI alone can trigger reduced size
                desired_signal = -REDUCED_SIZE
        
        # === NORMAL VOL MEAN REVERSION (no spike, but BB extreme) ===
        if not vol_spike:
            # Long: price below BB + RSI oversold + trend support
            if price_below_bb and rsi_oversold:
                if trend_bullish or above_sma200:
                    desired_signal = REDUCED_SIZE
            
            # Short: price above BB + RSI overbought + trend support
            if price_above_bb and rsi_overbought:
                if trend_bearish or below_sma200:
                    desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if mean reversion not complete ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if price hasn't returned to BB mid and vol still elevated
                if close[i] < bb_mid[i] and atr_ratio > 1.3:
                    desired_signal = BASE_SIZE
                # Hold if trend still bullish
                elif trend_bullish and rsi_4h[i] < 65:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if price hasn't returned to BB mid and vol still elevated
                if close[i] > bb_mid[i] and atr_ratio > 1.3:
                    desired_signal = -BASE_SIZE
                # Hold if trend still bearish
                elif trend_bearish and rsi_4h[i] > 35:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if vol normalized + price returned to BB mid
            if vol_normalized and close[i] > bb_mid[i]:
                desired_signal = 0.0
            # Exit if RSI extremely overbought
            if rsi_4h[i] > 75:
                desired_signal = 0.0
            # Exit if trend reverses strongly
            if trend_bearish and below_sma50:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if vol normalized + price returned to BB mid
            if vol_normalized and close[i] < bb_mid[i]:
                desired_signal = 0.0
            # Exit if RSI extremely oversold
            if rsi_4h[i] < 25:
                desired_signal = 0.0
            # Exit if trend reverses strongly
            if trend_bullish and above_sma50:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals