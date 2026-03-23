#!/usr/bin/env python3
"""
Experiment #1247: 1d Primary + 1w HTF — Dual Regime (Chop + Connors RSI)

Hypothesis: Research shows Choppiness Index + Connors RSI achieved ETH Sharpe +0.923.
Key insight: BTC/ETH 2025 is bear/range market, not trending. Need regime-adaptive logic:
1. CHOP(14) > 61.8 = range regime → use Connors RSI mean reversion
2. CHOP(14) < 38.2 = trend regime → use HMA crossover + RSI pullback
3. 1w HMA(21) for macro bias (only long if price > 1w HMA, only short if <)
4. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
5. Entry: CRSI < 15 (long) or CRSI > 85 (short) in range regime
6. Position size: 0.28 (discrete), ATR 2.5x trailing stop

Timeframe: 1d (target 30-60 trades/year = ~1 trade per week)
Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_crsi_1w_hma_atr_v1"
timeframe = "1d"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market consolidation vs trending
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean reversion signals
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Entry: CRSI < 10-15 (long), CRSI > 85-90 (short)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to positive values for RSI calculation
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs + 1e-10, streak_period)
    
    # Component 3: Percentile Rank of close over last 100 bars
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine all three components
    valid_mask = (~np.isnan(rsi_close)) & (~np.isnan(streak_rsi)) & (~np.isnan(percent_rank))
    crsi[valid_mask] = (rsi_close[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Discrete position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Signal persistence buffer
    signal_buffer = 0
    last_signal = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]) or atr[i] <= 1e-10:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(rsi_14[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_range = chop[i] > 55.0  # Range/choppy market
        regime_trend = chop[i] < 45.0  # Trending market
        
        # === DUAL REGIME LOGIC ===
        desired_signal = 0.0
        
        if regime_range:
            # MEAN REVERSION in range regime using Connors RSI
            # Long: CRSI oversold + macro not strongly bear
            if crsi[i] < 20.0 and not macro_bear:
                desired_signal = BASE_SIZE
            # Short: CRSI overbought + macro not strongly bull
            elif crsi[i] > 80.0 and not macro_bull:
                desired_signal = -BASE_SIZE
        
        elif regime_trend:
            # TREND FOLLOWING in trend regime using HMA + RSI pullback
            hma_bull = hma_21[i] > hma_50[i]
            hma_bear = hma_21[i] < hma_50[i]
            
            # Long: HMA bull + RSI pullback + macro bull
            if hma_bull and rsi_14[i] > 40.0 and rsi_14[i] < 60.0 and macro_bull:
                desired_signal = BASE_SIZE
            # Short: HMA bear + RSI pullback + macro bear
            elif hma_bear and rsi_14[i] > 40.0 and rsi_14[i] < 60.0 and macro_bear:
                desired_signal = -BASE_SIZE
        
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
        
        # === SIGNAL PERSISTENCE (prevent noise) ===
        if desired_signal != last_signal:
            signal_buffer += 1
            if signal_buffer >= 1:  # Only 1 bar persistence on 1d
                last_signal = desired_signal
                signal_buffer = 0
        else:
            signal_buffer = 0
        
        final_signal = last_signal
        
        # === DISCRETIZE SIGNAL VALUES ===
        if final_signal > 0:
            final_signal = BASE_SIZE
        elif final_signal < 0:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
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
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals