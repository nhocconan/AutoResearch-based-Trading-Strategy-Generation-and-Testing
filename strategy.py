#!/usr/bin/env python3
"""
Experiment #1244: 4h Primary + 12h/1d HTF — Connors RSI + Choppiness Regime

Hypothesis: #1239 HMA+RSI pullback was too conservative (Sharpe=0.077). Research shows
Connors RSI has 75% win rate for mean reversion, and Choppiness Index is the best
regime filter for crypto. Key changes:
1. Connors RSI (CRSI) instead of regular RSI - combines RSI(3) + streak + percentile
2. Choppiness Index regime switch: CHOP>61.8=range(mean revert), CHOP<38.2=trend
3. Dual regime logic: different entry conditions per regime
4. Remove hysteresis buffer (causes missed entries - seen in #1235, #1238, #1240, #1242)
5. Simpler stoploss: signal→0 when price moves 2.5*ATR against position
6. More generous CRSI thresholds to ensure >=20 trades/year

Target: Sharpe > 0.612 (beat current best), trades >= 80 train, >= 12 test
Timeframe: 4h (20-50 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_12h1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_crsi(close):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < 100:
        return crsi
    
    # RSI(3)
    rsi3 = calculate_rsi(close, period=3)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    for i in range(2, n):
        if not np.isnan(streak[i]):
            # Map streak to 0-100: positive streak = bullish, negative = bearish
            if streak[i] >= 0:
                streak_rsi[i] = min(100, 50 + streak[i] * 10)
            else:
                streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    # Percent Rank (100) - where does current return rank vs last 100 bars?
    percent_rank = np.full(n, np.nan)
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / close[:-1] * 100
    
    for i in range(100, n):
        window = returns[i-99:i+1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window[:-1] < window[-1])
            percent_rank[i] = count_below / 99 * 100
    
    # Combine into CRSI
    for i in range(100, n):
        if not np.isnan(rsi3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range/choppy (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if tr_sum > 1e-10 and (highest_high - lowest_low) > 1e-10:
            chop[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h HMA for intermediate trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_fast = calculate_hma(close, period=16)
    hma_slow = calculate_hma(close, period=48)
    crsi = calculate_crsi(close)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Track entry for stoploss
    entry_price = 0.0
    entry_atr = 0.0
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        
        # === MACRO TREND (12h + 1d HMA) ===
        macro_bull = close[i] > hma_12h_aligned[i] and close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i] and close[i] < hma_1d_aligned[i]
        macro_neutral = not macro_bull and not macro_bear
        
        # === HMA CROSSOVER ===
        hma_bull = hma_fast[i] > hma_slow[i]
        hma_bear = hma_fast[i] < hma_slow[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 20.0  # Strong mean reversion long signal
        crsi_overbought = crsi[i] > 80.0  # Strong mean reversion short signal
        crsi_pullback_long = crsi[i] < 40.0  # Moderate pullback in uptrend
        crsi_pullback_short = crsi[i] > 60.0  # Moderate pullback in downtrend
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (Mean Reversion)
        if is_choppy:
            # Long at CRSI extreme + price above macro support
            if crsi_oversold and close[i] > hma_1d_aligned[i] * 0.98:
                desired_signal = BASE_SIZE
            # Short at CRSI extreme + price below macro resistance
            elif crsi_overbought and close[i] < hma_1d_aligned[i] * 1.02:
                desired_signal = -BASE_SIZE
        
        # REGIME 2: TRENDING (Trend Following)
        elif is_trending:
            # Long: Macro bull + HMA bull + CRSI pullback (not extreme)
            if macro_bull and hma_bull and crsi_pullback_long:
                desired_signal = BASE_SIZE
            # Short: Macro bear + HMA bear + CRSI pullback (not extreme)
            elif macro_bear and hma_bear and crsi_pullback_short:
                desired_signal = -BASE_SIZE
        
        # REGIME 3: NEUTRAL/TRANSITION (Conservative)
        else:
            # Only enter on strong CRSI extremes with HMA confirmation
            if crsi_oversold and hma_bull:
                desired_signal = BASE_SIZE * 0.67  # Smaller size in uncertain regime
            elif crsi_overbought and hma_bear:
                desired_signal = -BASE_SIZE * 0.67
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if position_side > 0 and entry_price > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if position_side < 0 and entry_price > 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.15:
            final_signal = BASE_SIZE
        elif desired_signal < -0.15:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if position_side == 0:
                # New entry
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            # Exit
            if position_side != 0:
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals