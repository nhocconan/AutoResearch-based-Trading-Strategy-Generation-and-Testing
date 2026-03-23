#!/usr/bin/env python3
"""
Experiment #1097: 1d Primary + 1w HTF — KAMA Adaptive Trend with RSI Pullback

Hypothesis: After 794+ failed experiments, key insights for 1d timeframe:
1. Daily timeframe naturally generates 20-50 trades/year — optimal frequency
2. KAMA (Kaufman Adaptive Moving Average) adapts to volatility regimes better than HMA/EMA
3. 1w HTF provides macro trend filter without over-complication
4. RSI pullback entries (35/65 thresholds) ensure adequate trade frequency
5. Simple ATR trailing stop (2.5x) protects against large drawdowns
6. Discrete position sizes (0.25/0.30) minimize fee churn

Why this should beat Sharpe=0.612 (current best 4h strategy):
- 1d has cleaner signals than 4h, less noise and whipsaw
- KAMA adapts to BTC/ETH volatility regimes (2022 crash vs 2021 bull)
- 1w trend filter prevents counter-trend trades in strong moves
- RSI pullback entries catch both bull market dips and bear market rallies
- Proven pattern: KAMA + RSI + ATR worked on ETH (Sharpe +0.755 in research)

Timeframe: 1d (primary)
HTF: 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base, 0.15 reduced (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 20-50 trades/year, Sharpe > 0.612, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average — adapts to market noise.
    
    Formula:
    1. Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    2. Smoothing Constant (SC) = [ER * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)]^2
    3. KAMA[i] = KAMA[i-1] + SC * (Close[i] - KAMA[i-1])
    
    KAMA flattens in choppy markets, follows closely in trending markets.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < slow_period + efficiency_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(efficiency_period, n):
        signal = abs(close[i] - close[i - efficiency_period])
        noise = np.sum(np.abs(np.diff(close[i - efficiency_period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = np.full(n, np.nan)
    for i in range(efficiency_period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA with SMA of first slow_period bars
    kama[slow_period - 1] = np.mean(close[:slow_period])
    
    # Calculate KAMA
    for i in range(slow_period, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
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

def calculate_sma(close, period=50):
    """Simple Moving Average for trend filter."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w KAMA for macro trend filter
    kama_1w_raw = calculate_kama(df_1w['close'].values, efficiency_period=10, fast_period=2, slow_period=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate primary (1d) indicators
    kama_1d = calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30)
    rsi_1d = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after 200 SMA is ready
        # Skip if indicators not ready
        if np.isnan(kama_1d[i]) or np.isnan(rsi_1d[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(kama_1w_aligned[i]) or np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1w KAMA) ===
        # Weekly KAMA defines the primary trend direction
        macro_bull = close[i] > kama_1w_aligned[i]
        macro_bear = close[i] < kama_1w_aligned[i]
        
        # === PRIMARY TREND (1d KAMA) ===
        # Daily KAMA for entry timing
        kama_bull = close[i] > kama_1d[i]
        kama_bear = close[i] < kama_1d[i]
        
        # === SMA FILTER (long-term trend) ===
        # Only long above 200 SMA, only short below 200 SMA
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK (loose thresholds for trade frequency) ===
        # Long: RSI pulled back but not oversold (35-50 range)
        # Short: RSI rallied but not overbought (50-65 range)
        rsi_pullback_long = 35.0 <= rsi_1d[i] <= 50.0
        rsi_pullback_short = 50.0 <= rsi_1d[i] <= 65.0
        
        # === VOLATILITY CHECK ===
        # Avoid trading when ATR is extremely low (dead market)
        atr_ok = atr[i] > np.nanmedian(atr[max(0, i-100):i]) * 0.5
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY ===
        # Macro bull (1w) + Daily bull + Above 200 SMA + RSI pullback
        if macro_bull and kama_bull and above_sma200 and rsi_pullback_long and atr_ok:
            desired_signal = current_size
        
        # === SHORT ENTRY ===
        # Macro bear (1w) + Daily bear + Below 200 SMA + RSI pullback
        elif macro_bear and kama_bear and below_sma200 and rsi_pullback_short and atr_ok:
            desired_signal = -current_size
        
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
                # Hold long if macro and daily still bull
                if macro_bull and kama_bull:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if macro and daily still bear
                if macro_bear and kama_bear:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses or RSI overbought
            if macro_bear or rsi_1d[i] > 70.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses or RSI oversold
            if macro_bull or rsi_1d[i] < 30.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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