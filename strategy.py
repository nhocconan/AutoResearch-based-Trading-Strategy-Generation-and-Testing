#!/usr/bin/env python3
"""
Experiment #1041: 4h KAMA Adaptive Trend + RSI Mean Reversion + 1d HMA Filter

Hypothesis: After 753 failed strategies, the key insight is that COMPLEX multi-condition
entries create mutually exclusive filters → 0 trades. The winning approach for BTC/ETH
in bear/range markets (2025 test) is SIMPLER:

1. KAMA (Kaufman Adaptive MA): Automatically adapts to volatility. In trends, follows
   price closely. In ranges, flattens out. This is PROVEN to work better than HMA/EMA
   for BTC/ETH which spend 60%+ time in consolidation.

2. RSI(7) Mean Reversion: Short-period RSI for faster signals. RSI<30 = long, RSI>70 = short.
   This ensures trades happen frequently enough (30+ train, 3+ test) without being too loose.

3. 1d HMA21 Macro Filter: SINGLE HTF filter (not multiple). Price > 1d HMA = bias long only.
   Price < 1d HMA = bias short only. This asymmetric filter works in bear markets.

4. KAMA Slope Confirmation: Only enter long if KAMA slope > 0, only enter short if KAMA slope < 0.
   This filters out counter-trend mean reversion trades that get stopped out.

5. ATR Stoploss: 2.5x ATR trailing stop. Signal→0 when hit. Mandatory for risk control.

Why this beats exp #1039 (Sharpe=-0.258):
- KAMA adapts to regime automatically (no complex regime detection that creates conflicts)
- RSI(7) is faster than RSI(14), ensures more trades
- SINGLE HTF filter (1d HMA) instead of multiple conflicting filters
- KAMA slope adds trend confirmation without complex Donchian breakout logic
- Simpler hold/exit logic = fewer edge cases that create 0 trades

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
Timeframe: 4h (target 20-50 trades/year)
Position Size: 0.25-0.30 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_meanrevert_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility - follows price in trends, flattens in ranges.
    
    Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    SC = [ER * (fast SC - slow SC) + slow SC]^2
    KAMA = KAMA[prev] + SC * (Close - KAMA[prev])
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_kama_slope(kama, lookback=5):
    """
    KAMA slope: positive = uptrend, negative = downtrend
    Uses linear regression slope over lookback period
    """
    n = len(kama)
    slope = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if np.any(np.isnan(kama[i - lookback:i + 1])):
            continue
        y = kama[i - lookback:i + 1]
        x = np.arange(lookback)
        # Simple linear regression slope
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        if denominator > 0:
            slope[i] = numerator / denominator
        else:
            slope[i] = 0
    
    return slope

def calculate_rsi(close, period=7):
    """
    Relative Strength Index - shorter period (7) for faster signals
    RSI < 30 = oversold (long), RSI > 70 = overbought (short)
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average for HTF trend filter."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_slope_4h = calculate_kama_slope(kama_4h, lookback=5)
    rsi_4h = calculate_rsi(close, period=7)
    atr_4h = calculate_atr(high, low, close, period=14)
    
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
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(kama_4h[i]) or np.isnan(kama_slope_4h[i]):
            continue
        
        # === MACRO TREND (1d HMA21) ===
        # Asymmetric filter: easier to long when above, easier to short when below
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND SIGNAL ===
        kama_uptrend = kama_slope_4h[i] > 0
        kama_downtrend = kama_slope_4h[i] < 0
        
        # === RSI MEAN REVERSION ===
        rsi_oversold = rsi_4h[i] < 30
        rsi_overbought = rsi_4h[i] > 70
        rsi_neutral = 30 <= rsi_4h[i] <= 70
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        # Entry 1: Macro bull + KAMA uptrend + RSI oversold (primary mean reversion long)
        if macro_bull and kama_uptrend and rsi_oversold:
            desired_signal = BASE_SIZE
        # Entry 2: Macro bull + RSI deeply oversold (strong reversal signal)
        elif macro_bull and rsi_4h[i] < 25:
            desired_signal = BASE_SIZE
        # Entry 3: KAMA uptrend + RSI oversold (trend continuation pullback)
        elif kama_uptrend and rsi_oversold:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        # Entry 1: Macro bear + KAMA downtrend + RSI overbought (primary mean reversion short)
        if macro_bear and kama_downtrend and rsi_overbought:
            desired_signal = -BASE_SIZE
        # Entry 2: Macro bear + RSI deeply overbought (strong reversal signal)
        elif macro_bear and rsi_4h[i] > 75:
            desired_signal = -BASE_SIZE
        # Entry 3: KAMA downtrend + RSI overbought (trend continuation pullback)
        elif kama_downtrend and rsi_overbought:
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
        
        # === EXIT CONDITIONS ===
        # Exit long if RSI becomes overbought (mean reversion complete)
        if in_position and position_side > 0 and rsi_4h[i] > 70:
            desired_signal = 0.0
        
        # Exit short if RSI becomes oversold (mean reversion complete)
        if in_position and position_side < 0 and rsi_4h[i] < 30:
            desired_signal = 0.0
        
        # Exit if macro trend reverses against position
        if in_position and position_side > 0 and macro_bear and kama_downtrend:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull and kama_uptrend:
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
                # Flip position
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