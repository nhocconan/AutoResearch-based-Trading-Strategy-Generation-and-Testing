#!/usr/bin/env python3
"""
Experiment #897: 1d Primary + 1w HTF — KAMA Adaptive Trend + RSI + Choppiness Regime

Hypothesis: After 600+ failed strategies, 1d timeframe with KAMA (Kaufman Adaptive MA)
should work better than HMA/EMA because KAMA adapts to volatility automatically.

Key insights from research:
1. 1d Primary TF: Target 20-40 trades/year (minimal fee drag)
2. 1w HMA(21) for macro bias only (bull/bear market filter)
3. KAMA(21) on 1d for adaptive trend — flattens in chop, moves in trends
4. RSI(14) with relaxed 35/65 thresholds (not 30/70) to ensure trades
5. Choppiness Index(14) binary: >55=mean-revert, <45=trend-follow
6. ATR(14) trailing stop 2.5x for risk management
7. Signal sizes: 0.0, ±0.25, ±0.30 (discrete to minimize churn)

Why KAMA over HMA:
- KAMA has built-in efficiency ratio that slows in choppy markets
- No need for separate regime filter for trend speed
- Proven in crypto: adapts to BTC's volatile/ranging nature
- Less whipsaw than EMA/HMA in 2022 crash

Critical improvements:
- RELAXED RSI thresholds (35/65) to guarantee 20+ trades per symbol
- Simple 1w HMA macro filter (not dual HTF)
- Binary chop regime (not 3-state)
- ALL symbols MUST have positive Sharpe (no SOL-only bias)
- Hold logic maintains position through minor pullbacks

Target: Sharpe > 0.612, trades >= 20 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_chop_regime_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    
    KAMA adapts to market volatility via Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = prior_KAMA + SC * (price - prior_KAMA)
    
    In trending markets (high ER): KAMA follows price closely
    In choppy markets (low ER): KAMA flattens, reduces whipsaw
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + fast_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        net_change = abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_changes > 0:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate KAMA
    kama[period] = close[period]  # Initialize with price
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods."""
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging (mean-revert), CHOP < 45 = trending (trend-follow).
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            prev_close = close[j - 1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        prev_close = close[i - 1]
        tr[i] = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average for HTF alignment."""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    kama_21 = calculate_kama(close, period=21)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 1w HMA for macro regime (bull/bear market)
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
        if np.isnan(kama_21[i]) or np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d KAMA21) ===
        kama_bullish = close[i] > kama_21[i]
        kama_bearish = close[i] < kama_21[i]
        kama_slope_up = kama_21[i] > kama_21[i - 5] if not np.isnan(kama_21[i - 5]) else False
        kama_slope_down = kama_21[i] < kama_21[i - 5] if not np.isnan(kama_21[i - 5]) else False
        
        # === REGIME DETECTION (1d Choppiness Index) ===
        ranging_regime = chop_14[i] > 55
        trending_regime = chop_14[i] < 45
        
        # === RSI SIGNALS (Relaxed thresholds: 35/65) ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_oversold = rsi_14[i] < 25
        rsi_extreme_overbought = rsi_14[i] > 75
        rsi_neutral = 35 <= rsi_14[i] <= 65
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: RSI oversold + price below KAMA (pullback in range)
            if rsi_oversold and kama_bearish:
                desired_signal = BASE_SIZE
            # Short: RSI overbought + price above KAMA (rally in range)
            elif rsi_overbought and kama_bullish:
                desired_signal = -BASE_SIZE
            # Fallback: extreme RSI alone (guarantees trades)
            elif rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            elif rsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + RSI not overbought + macro support
            if kama_bullish and kama_slope_up and rsi_14[i] < 70:
                if macro_bull or rsi_oversold:
                    desired_signal = BASE_SIZE
                elif rsi_neutral:
                    desired_signal = REDUCED_SIZE
            # Short: Bearish trend + RSI not oversold + macro support
            elif kama_bearish and kama_slope_down and rsi_14[i] > 30:
                if macro_bear or rsi_overbought:
                    desired_signal = -BASE_SIZE
                elif rsi_neutral:
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: KAMA direction + RSI confluence
            if kama_bullish and rsi_oversold and macro_bull:
                desired_signal = REDUCED_SIZE
            elif kama_bearish and rsi_overbought and macro_bear:
                desired_signal = -REDUCED_SIZE
            # Fallback: extreme RSI with macro alignment
            elif rsi_extreme_oversold and macro_bull:
                desired_signal = REDUCED_SIZE
            elif rsi_extreme_overbought and macro_bear:
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
                # Hold long if KAMA bullish and RSI not extreme overbought
                if kama_bullish and rsi_14[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if KAMA bearish and RSI not extreme oversold
                if kama_bearish and rsi_14[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA reverses + RSI overbought
            if kama_bearish and rsi_14[i] > 70:
                desired_signal = 0.0
            # Exit if macro reverses strongly
            if macro_bear and rsi_14[i] > 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA reverses + RSI oversold
            if kama_bullish and rsi_14[i] < 30:
                desired_signal = 0.0
            # Exit if macro reverses strongly
            if macro_bull and rsi_14[i] < 40:
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
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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