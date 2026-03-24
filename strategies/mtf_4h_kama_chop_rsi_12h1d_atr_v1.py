#!/usr/bin/env python3
"""
Experiment #794: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + Choppiness Regime + RSI Entry

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. 4h timeframe balances trade frequency (20-50/year) with signal quality
2. KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than EMA/HMA
3. 1d Choppiness Index provides regime detection (ranging vs trending)
4. 12h HMA(21) gives smoother trend bias than 1w (more responsive to regime changes)
5. RSI(14) with relaxed thresholds (30/70) generates sufficient trades without overtrading
6. ATR(14) trailing stop at 2.5x protects from major drawdowns
7. Dual regime logic: mean revert when CHOP>55, trend follow when CHOP<45
8. Position sizing: 0.25-0.30 discrete levels to control fees

Strategy design:
1. 12h HMA(21) for medium-term trend bias (aligned via mtf_data helper)
2. 1d Choppiness Index(14) for regime detection
3. 4h KAMA(10,2,30) for adaptive trend following
4. 4h RSI(14) for entry timing
5. 4h ATR(14) for trailing stop (2.5x)
6. 4h Bollinger Bands(20, 2.0) for mean reversion bounds
7. Discrete signals: 0.0, ±0.25, ±0.30
8. Relaxed entry thresholds to ensure >=10 trades/train, >=3 trades/test

Key differences from failed 4h strategies:
- KAMA instead of EMA/HMA for primary trend (adapts to volatility)
- 12h HMA for trend bias (not 1w - too slow for 4h entries)
- RSI thresholds: 30/70 (not 20/80) — generates more trades
- CHOP thresholds: 55/45 — more regime switches
- Hold logic: maintain position until opposite signal or stoploss
- Volume filter: 1.3x (not 1.5x) — less restrictive

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_chop_rsi_12h1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_ema(series, period):
    """Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values

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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise - moves fast in trends, slow in chop.
    ER (Efficiency Ratio) = |close - close_n| / sum(|close_i - close_i-1|)
    SC (Smoothing Constant) = ER * (fast_sc - slow_sc) + slow_sc
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < slow_period + er_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = er[i] * (fast_sc - slow_sc) + slow_sc
        kama[i] = kama[i - 1] + sc * sc * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    sma = calculate_sma(close, period)
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending.
    We use 55/45 for more regime switches on 4h.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_mult=2.0)
    vol_sma_4h = calculate_volume_sma(volume, period=20)
    
    # Calculate and align HTF HMA for trend bias (12h)
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d Choppiness for regime detection
    chop_1d_raw = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(bb_sma[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(chop_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_4h[i]) or vol_sma_4h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (12h HTF HMA21) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === REGIME DETECTION (1d Choppiness Index) ===
        ranging_regime = chop_1d_aligned[i] > 55
        trending_regime = chop_1d_aligned[i] < 45
        neutral_regime = not ranging_regime and not trending_regime
        
        # === VOLUME CONFIRMATION (relaxed) ===
        volume_confirmed = volume[i] > 1.3 * vol_sma_4h[i]
        
        # === RSI SIGNALS (relaxed thresholds for more trades) ===
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        rsi_extreme_oversold = rsi_4h[i] < 25
        rsi_extreme_overbought = rsi_4h[i] > 75
        rsi_neutral_low = 35 < rsi_4h[i] < 50
        rsi_neutral_high = 50 < rsi_4h[i] < 65
        
        # === BOLLINGER POSITION ===
        below_bb_lower = close[i] < bb_lower[i]
        above_bb_upper = close[i] > bb_upper[i]
        near_bb_lower = close[i] < bb_sma[i] * 0.98
        near_bb_upper = close[i] > bb_sma[i] * 1.02
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) ===
        if ranging_regime:
            # Mean reversion long: RSI oversold + below BB lower + 12h bullish
            if rsi_oversold and below_bb_lower and trend_12h_bullish:
                desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Mean reversion short: RSI overbought + above BB upper + 12h bearish
            if rsi_overbought and above_bb_upper and trend_12h_bearish:
                desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Conservative: extreme RSI + KAMA alignment
            if rsi_extreme_oversold and kama_bullish:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and kama_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) ===
        elif trending_regime:
            # Trend pullback long: 12h bullish + KAMA bullish + RSI neutral low
            if trend_12h_bullish and kama_bullish and rsi_neutral_low:
                desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Trend pullback short: 12h bearish + KAMA bearish + RSI neutral high
            if trend_12h_bearish and kama_bearish and rsi_neutral_high:
                desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Breakout continuation with volume
            if trend_12h_bullish and kama_bullish and above_bb_upper and volume_confirmed:
                desired_signal = BASE_SIZE
            
            if trend_12h_bearish and kama_bearish and below_bb_lower and volume_confirmed:
                desired_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: only extreme RSI + trend alignment
            if rsi_extreme_oversold and trend_12h_bullish and kama_bullish:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and trend_12h_bearish and kama_bearish:
                desired_signal = -REDUCED_SIZE
            
            # Also allow basic mean reversion with volume
            if rsi_oversold and below_bb_lower and trend_12h_bullish:
                desired_signal = REDUCED_SIZE if volume_confirmed else 0.0
            
            if rsi_overbought and above_bb_upper and trend_12h_bearish:
                desired_signal = -REDUCED_SIZE if volume_confirmed else 0.0
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and RSI not overbought
                if trend_12h_bullish and kama_bullish and rsi_4h[i] < 75:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if trend intact and RSI not oversold
                if trend_12h_bearish and kama_bearish and rsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or RSI overbought
            if trend_12h_bearish and rsi_4h[i] > 70:
                desired_signal = 0.0
            # Exit if price hits BB upper in ranging regime
            if ranging_regime and above_bb_upper:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or RSI oversold
            if trend_12h_bullish and rsi_4h[i] < 30:
                desired_signal = 0.0
            # Exit if price hits BB lower in ranging regime
            if ranging_regime and below_bb_lower:
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