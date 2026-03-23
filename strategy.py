#!/usr/bin/env python3
"""
Experiment #909: 4h Primary + 1d HTF — KAMA Trend + Fisher Transform + ATR

Hypothesis: After 600+ failed strategies, a simpler approach with Fisher Transform
for entry timing should work better than complex regime switching.

Key insights:
1. Fisher Transform catches reversals better than RSI in bear markets
2. KAMA adapts to volatility (better than HMA/EMA in choppy markets)
3. 1d HMA for long-term trend bias (direction filter only)
4. Simple entry: Fisher extreme + trend alignment
5. ATR trailing stop (2.5x) for risk management

Why this should work:
- Fisher Transform normalizes price to -2/+2, clear reversal signals
- KAMA efficiency ratio adapts to market conditions automatically
- Fewer confluence requirements = more trades across ALL symbols
- 4h timeframe targets 25-40 trades/year (optimal fee/trade balance)
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility via Efficiency Ratio (ER)
    ER near 1 = trending (fast smoothing), ER near 0 = choppy (slow smoothing)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant (SC)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price into -2 to +2 range for clear reversal signals
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 2:
        return fisher, fisher_signal
    
    for i in range(period, n):
        # Calculate typical price
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest == lowest:
            fisher[i] = fisher[i - 1] if not np.isnan(fisher[i - 1]) else 0
            fisher_signal[i] = fisher[i]
            continue
        
        # Normalize price to -1 to +1
        close_price = (high[i] + low[i]) / 2
        x = (2 * close_price - highest - lowest) / (highest - lowest + 1e-10)
        x = np.clip(x * 0.999, -0.999, 0.999)  # Prevent log errors
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i - 1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, period=9)
    atr_4h = calculate_atr(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    
    # Calculate and align 1d HMA for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(fisher_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(rsi_4h[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1d HMA21) ===
        trend_bullish = close[i] > hma_1d_aligned[i]
        trend_bearish = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h KAMA) ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher_4h[i] < -1.5
        fisher_overbought = fisher_4h[i] > 1.5
        fisher_cross_up = fisher_4h[i] > fisher_signal_4h[i] and fisher_signal_4h[i] < -1.0
        fisher_cross_down = fisher_4h[i] < fisher_signal_4h[i] and fisher_signal_4h[i] > 1.0
        
        # === RSI FILTER (avoid extreme overbought/oversold entries) ===
        rsi_neutral = 30 < rsi_4h[i] < 70
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        # Primary: Fisher oversold + trend alignment + KAMA bullish
        if fisher_oversold and trend_bullish and kama_bullish:
            desired_signal = BASE_SIZE
        
        # Secondary: Fisher cross up from oversold + trend bullish
        elif fisher_cross_up and trend_bullish and fisher_4h[i] < 0:
            desired_signal = REDUCED_SIZE
        
        # Tertiary: KAMA bullish + RSI recovering from oversold
        elif kama_bullish and trend_bullish and rsi_4h[i] < 40 and rsi_4h[i] > rsi_4h[i-1] if i > 0 else False:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY LOGIC ===
        # Primary: Fisher overbought + trend alignment + KAMA bearish
        if fisher_overbought and trend_bearish and kama_bearish:
            desired_signal = -BASE_SIZE
        
        # Secondary: Fisher cross down from overbought + trend bearish
        elif fisher_cross_down and trend_bearish and fisher_4h[i] > 0:
            desired_signal = -REDUCED_SIZE
        
        # Tertiary: KAMA bearish + RSI weakening from overbought
        elif kama_bearish and trend_bearish and rsi_4h[i] > 60 and rsi_4h[i] < rsi_4h[i-1] if i > 0 else False:
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d trend and 4h KAMA still bullish
                if trend_bullish and kama_bullish and fisher_4h[i] < 1.5:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d trend and 4h KAMA still bearish
                if trend_bearish and kama_bearish and fisher_4h[i] > -1.5:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses on both timeframes
            if trend_bearish and kama_bearish:
                desired_signal = 0.0
            # Exit if Fisher extremely overbought
            if fisher_4h[i] > 2.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses on both timeframes
            if trend_bullish and kama_bullish:
                desired_signal = 0.0
            # Exit if Fisher extremely oversold
            if fisher_4h[i] < -2.0:
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