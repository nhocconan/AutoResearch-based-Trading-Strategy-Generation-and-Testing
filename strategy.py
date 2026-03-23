#!/usr/bin/env python3
"""
Experiment #806: 12h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Filter + ATR Trail

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. 12h timeframe produces 20-50 trades/year — optimal for fee minimization
2. Donchian(20) breakout captures major trend moves on higher TF
3. 1d HMA(21) provides cleaner trend bias than 12h (smoother, less noise)
4. RSI(14) filter at 50 level ensures momentum confirmation on breakouts
5. ATR(14) trailing stop at 3.0x for 12h (wider stops needed on higher TF)
6. Volume filter 1.5x ensures breakout has participation
7. Dual regime: trend-follow breakouts only, no mean reversion on 12h
8. Position sizing: 0.30-0.35 discrete levels to control fees and drawdown

Why 12h works better than 4h for BTC/ETH:
- 4h has too many false breakouts in 2022-2024 chop
- 12h filters out noise, only captures major moves
- 2025 bear market needs fewer, higher-quality trades
- Historical 12h strategies showed Sharpe +0.78 to +0.88 on SOL

Strategy design:
1. 1d HMA(21) for long-term trend bias (aligned via mtf_data helper)
2. 12h Donchian(20) for breakout detection
3. 12h HMA(16) for short-term trend confirmation
4. 12h RSI(14) for momentum filter (>50 for long, <50 for short)
5. 12h ATR(14) for trailing stop (3.0x)
6. 12h Volume SMA(20) for breakout confirmation (1.5x)
7. Discrete signals: 0.0, ±0.30, ±0.35
8. Strict entry: ALL conditions must align (trend + breakout + RSI + volume)

Key differences from failed strategies:
- 12h instead of 4h (fewer trades, higher quality)
- Donchian breakout instead of EMA crossover (proven on 12h)
- 1d HMA for trend (not 12h HMA — smoother bias)
- RSI >50/<50 filter (not extreme levels — ensures momentum)
- 3.0x ATR stop (wider for 12h volatility)
- NO mean reversion logic (12h is pure trend-follow)

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_1d_rsi_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average — smoother and more responsive than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

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

def calculate_donchian(high, low, period=20):
    """Donchian Channels — breakout detection."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
        mid[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, mid

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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    hma_12h_16 = calculate_hma(close, 16)
    hma_12h_48 = calculate_hma(close, 48)
    rsi_12h = calculate_rsi(close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    vol_sma_12h = calculate_volume_sma(volume, period=20)
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.35
    REDUCED_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            continue
        if np.isnan(rsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_12h[i]) or vol_sma_12h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA TREND CONFIRMATION ===
        hma_12h_bullish = hma_12h_16[i] > hma_12h_48[i]
        hma_12h_bearish = hma_12h_16[i] < hma_12h_48[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.5 * vol_sma_12h[i]
        
        # === RSI MOMENTUM FILTER ===
        rsi_bullish = rsi_12h[i] > 50
        rsi_bearish = rsi_12h[i] < 50
        rsi_strong_bull = rsi_12h[i] > 55
        rsi_strong_bear = rsi_12h[i] < 45
        
        # === DONCHIAN BREAKOUT DETECTION ===
        breakout_long = close[i] >= donch_upper[i]
        breakout_short = close[i] <= donch_lower[i]
        
        # === PRICE POSITION RELATIVE TO DONCHIAN ===
        near_donch_upper = close[i] > donch_mid[i] * 1.01
        near_donch_lower = close[i] < donch_mid[i] * 0.99
        
        desired_signal = 0.0
        
        # === LONG ENTRY: All conditions must align ===
        if trend_1d_bullish and hma_12h_bullish and rsi_bullish:
            # Primary: Donchian breakout with volume
            if breakout_long and volume_confirmed:
                desired_signal = BASE_SIZE
            # Secondary: Near breakout with strong RSI
            elif near_donch_upper and rsi_strong_bull:
                desired_signal = REDUCED_SIZE if volume_confirmed else 0.0
        
        # === SHORT ENTRY: All conditions must align ===
        if trend_1d_bearish and hma_12h_bearish and rsi_bearish:
            # Primary: Donchian breakdown with volume
            if breakout_short and volume_confirmed:
                desired_signal = -BASE_SIZE
            # Secondary: Near breakdown with strong RSI
            elif near_donch_lower and rsi_strong_bear:
                desired_signal = -REDUCED_SIZE if volume_confirmed else 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x for 12h) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d trend and 12h HMA intact
                if trend_1d_bullish and hma_12h_bullish and rsi_12h[i] > 40:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d trend and 12h HMA intact
                if trend_1d_bearish and hma_12h_bearish and rsi_12h[i] < 60:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses
            if trend_1d_bearish and hma_12h_bearish:
                desired_signal = 0.0
            # Exit if RSI extremely overbought
            if rsi_12h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses
            if trend_1d_bullish and hma_12h_bullish:
                desired_signal = 0.0
            # Exit if RSI extremely oversold
            if rsi_12h[i] < 20:
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
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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