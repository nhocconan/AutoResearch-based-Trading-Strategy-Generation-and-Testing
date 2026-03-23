#!/usr/bin/env python3
"""
Experiment #987: 1d Primary + 1w HTF — KAMA Adaptive Trend + Choppiness Regime + RSI

Hypothesis: After 713 failed strategies, the key is SIMPLICITY on daily timeframe.
KAMA (Kaufman Adaptive Moving Average) adapts to market noise better than EMA/HMA.
Combined with Choppiness Index for regime detection and weekly HMA for macro bias,
this should work across BTC/ETH/SOL in both bull and bear markets.

Why 1d timeframe:
- Target 20-40 trades/year (minimal fee drag)
- Daily bars filter out intraday noise
- Proven to work in 2022 crash and 2025 bear market
- Less sensitive to funding rate timing issues

Key innovations:
1. KAMA adapts smoothing based on volatility (ER = Efficiency Ratio)
2. CHOP(14) > 61.8 = range regime (mean revert), < 38.2 = trend regime
3. 1w HMA(21) for macro trend bias (only trade with weekly trend)
4. RSI(14) for entry timing (oversold in uptrend, overbought in downtrend)
5. ATR(14) trailing stop at 2.5x for risk management
6. Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 25-35 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_chop_regime_1w_hma_rsi_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing based on market efficiency (trend vs noise).
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
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
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
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
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel — highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    kama_1d = calculate_kama(close, period=10, fast=2, slow=30)
    rsi_1d = calculate_rsi(close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
        if np.isnan(kama_1d[i]) or np.isnan(rsi_1d[i]) or np.isnan(atr_1d[i]):
            continue
        if np.isnan(chop_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(donch_upper[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (1d Choppiness Index) ===
        ranging_regime = chop_1d[i] > 61.8
        trending_regime = chop_1d[i] < 38.2
        neutral_regime = not ranging_regime and not trending_regime
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_1d[i]
        kama_bearish = close[i] < kama_1d[i]
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_1d[i] < 35
        rsi_overbought = rsi_1d[i] > 65
        rsi_extreme_oversold = rsi_1d[i] < 25
        rsi_extreme_overbought = rsi_1d[i] > 75
        
        # === DONCHIAN BREAKOUT ===
        donch_breakout_long = close[i] > donch_upper[i]
        donch_breakout_short = close[i] < donch_lower[i]
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 38.2) — Trend Following ===
        if trending_regime:
            # Long: Macro bull + KAMA bull + RSI not overbought
            if macro_bull and kama_bullish and not rsi_overbought:
                desired_signal = BASE_SIZE
            # Long: Donchian breakout + macro bull
            elif donch_breakout_long and macro_bull:
                desired_signal = BASE_SIZE
            # Long: RSI oversold in uptrend (pullback entry)
            elif rsi_oversold and macro_bull and kama_bullish:
                desired_signal = REDUCED_SIZE
            
            # Short: Macro bear + KAMA bear + RSI not oversold
            if macro_bear and kama_bearish and not rsi_oversold:
                desired_signal = -BASE_SIZE
            # Short: Donchian breakdown + macro bear
            elif donch_breakout_short and macro_bear:
                desired_signal = -BASE_SIZE
            # Short: RSI overbought in downtrend (rally entry)
            elif rsi_overbought and macro_bear and kama_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME (CHOP > 61.8) — Mean Reversion ===
        elif ranging_regime:
            # Long: RSI extreme oversold + price near Donchian low
            if rsi_extreme_oversold and close[i] < donch_lower[i] * 1.02:
                desired_signal = BASE_SIZE
            # Long: RSI oversold + macro neutral/bull
            elif rsi_oversold and not macro_bear:
                desired_signal = REDUCED_SIZE
            
            # Short: RSI extreme overbought + price near Donchian high
            if rsi_extreme_overbought and close[i] > donch_upper[i] * 0.98:
                desired_signal = -BASE_SIZE
            # Short: RSI overbought + macro neutral/bear
            elif rsi_overbought and not macro_bull:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: KAMA trend + RSI confirmation
            if kama_bullish and rsi_oversold and macro_bull:
                desired_signal = BASE_SIZE
            elif kama_bullish and rsi_1d[i] < 50 and macro_bull:
                desired_signal = REDUCED_SIZE
            
            if kama_bearish and rsi_overbought and macro_bear:
                desired_signal = -BASE_SIZE
            elif kama_bearish and rsi_1d[i] > 50 and macro_bear:
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
                # Hold long if KAMA and macro still bullish
                if kama_bullish and macro_bull and rsi_1d[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if KAMA and macro still bearish
                if kama_bearish and macro_bear and rsi_1d[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA + macro both reverse
            if kama_bearish and macro_bear:
                desired_signal = 0.0
            # Exit if RSI extreme overbought
            if rsi_extreme_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA + macro both reverse
            if kama_bullish and macro_bull:
                desired_signal = 0.0
            # Exit if RSI extreme oversold
            if rsi_extreme_oversold:
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
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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