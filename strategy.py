#!/usr/bin/env python3
"""
Experiment #1127: 1d Primary + 1w HTF — Adaptive KAMA Regime with RSI Entries

Hypothesis: After analyzing 820+ failed experiments, key insights for 1d timeframe:
1. KAMA (Kaufman Adaptive) outperforms HMA/EMA in crypto's varying volatility regimes
2. Choppiness Index should GUIDE entries, not BLOCK them (loose thresholds)
3. RSI(7) faster than RSI(14) for daily entries — catches moves earlier
4. 1w KMA provides macro bias without over-filtering (just direction, not strength)
5. DUAL REGIME: trend-follow when CHOP<50, mean-revert when CHOP>55, transition zone flexible
6. LOOSE entry thresholds ensure 20-40 trades/year on 1d (critical for Sharpe)
7. Position size 0.25-0.30 with 2.5x ATR trailing stop

Why this should beat Sharpe=0.612:
- KAMA adapts efficiency ratio to market noise — better than fixed HMA
- Choppiness as regime guide (not hard filter) prevents 0-trade scenarios
- RSI(7) more responsive than RSI(14) on daily bars
- 1w macro filter only checks direction (price vs KAMA), not strength
- Works on BTC/ETH mean-reversion AND SOL trend patterns

Timeframe: 1d (primary)
HTF: 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.28 base, 0.15 reduced (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 20-40 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_chop_rsi_1w_regime_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average — adapts to market noise.
    
    Formula:
    1. Efficiency Ratio (ER) = |close - close[n]| / sum(|close[i] - close[i-1]|)
    2. Smoothing Constant (SC) = [ER * (fast_SC - slow_SC) + slow_SC]^2
    3. KAMA = KAMA_prev + SC * (close - KAMA_prev)
    
    ER near 1 = trending (fast response)
    ER near 0 = choppy (slow response)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = sum(abs(close[j] - close[j-1]) for j in range(i - period + 1, i + 1))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA with SMA of first period
    kama[period - 1] = np.mean(close[:period])
    
    # Calculate KAMA
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=7):
    """Relative Strength Index — faster period for daily entries."""
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market consolidation vs trending.
    
    Formula:
    CHOP = 100 * LOG10(sum(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range-bound
    CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR for each bar (simplified: just true range)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = sum(tr[i-period+1:i+1])
        highest = max(high[i-period+1:i+1])
        lowest = min(low[i-period+1:i+1])
        range_hl = highest - lowest
        
        if range_hl > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w KAMA for macro trend bias
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=34)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate primary (1d) indicators
    kama_1d = calculate_kama(close, period=21)
    rsi_1d = calculate_rsi(close, period=7)  # Faster RSI for daily
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.14
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_1d[i]) or np.isnan(rsi_1d[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(chop[i]) or np.isnan(kama_1w_aligned[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO BIAS (1w KAMA) ===
        # Simple direction filter — only checks if price above/below weekly KAMA
        macro_bull = close[i] > kama_1w_aligned[i]
        macro_bear = close[i] < kama_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        # LOOSE thresholds to avoid 0-trade scenario
        # CHOP < 50 = trending regime (follow KAMA direction)
        # CHOP > 55 = choppy regime (mean revert at RSI extremes)
        # 50-55 = transition (allow both)
        is_trending = chop[i] < 50.0
        is_choppy = chop[i] > 55.0
        
        # === DAILY KAMA DIRECTION ===
        kama_bull = close[i] > kama_1d[i]
        kama_bear = close[i] < kama_1d[i]
        
        # === RSI SIGNALS (faster 7-period) ===
        # Looser thresholds for more trades
        rsi_oversold = rsi_1d[i] < 40.0
        rsi_overbought = rsi_1d[i] > 60.0
        rsi_extreme_oversold = rsi_1d[i] < 30.0
        rsi_extreme_overbought = rsi_1d[i] > 70.0
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === TRENDING REGIME (CHOP < 50) ===
        # Follow KAMA direction with RSI pullback confirmation
        if is_trending:
            # Long: daily KAMA bull + RSI pullback (not extreme) + macro neutral/bull
            if kama_bull and rsi_oversold and not rsi_extreme_oversold:
                if macro_bull or not macro_bear:  # macro not strongly bear
                    desired_signal = current_size
            
            # Short: daily KAMA bear + RSI pullback (not extreme) + macro neutral/bear
            elif kama_bear and rsi_overbought and not rsi_extreme_overbought:
                if macro_bear or not macro_bull:  # macro not strongly bull
                    desired_signal = -current_size
        
        # === CHOPPY REGIME (CHOP > 55) ===
        # Mean reversion at RSI extremes
        elif is_choppy:
            # Long: RSI extreme oversold (mean revert up)
            if rsi_extreme_oversold:
                desired_signal = current_size
            
            # Short: RSI extreme overbought (mean revert down)
            elif rsi_extreme_overbought:
                desired_signal = -current_size
        
        # === TRANSITION ZONE (50 <= CHOP <= 55) ===
        # Allow either strategy with stricter confirmation
        else:
            # Long: need both KAMA bull AND RSI oversold
            if kama_bull and rsi_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short: need both KAMA bear AND RSI overbought
            elif kama_bear and rsi_overbought:
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if KAMA still bull or in choppy regime with RSI not overbought
                if kama_bull or (is_choppy and rsi_1d[i] < 65.0):
                    desired_signal = current_size if is_trending else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if KAMA still bear or in choppy regime with RSI not oversold
                if kama_bear or (is_choppy and rsi_1d[i] > 35.0):
                    desired_signal = -current_size if is_trending else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA reverses bear OR RSI extreme overbought
            if kama_bear and rsi_1d[i] > 55.0:
                desired_signal = 0.0
            elif rsi_extreme_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA reverses bull OR RSI extreme oversold
            if kama_bull and rsi_1d[i] < 45.0:
                desired_signal = 0.0
            elif rsi_extreme_oversold:
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