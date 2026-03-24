#!/usr/bin/env python3
"""
Experiment #659: 1h Primary + 4h/12h HTF — Regime-Adaptive CRSI + HMA Pullback

Hypothesis: 1h timeframe with regime detection (Choppiness) should generate MORE trades
than 4h/6h strategies while maintaining quality. Key insight from failures: too many
filters = 0 trades. This strategy uses LOOSE entry conditions with HTF bias filter.

Regime-adaptive logic:
- RANGE (CHOP > 55): Mean revert using Connors RSI extremes (CRSI < 20 long, > 80 short)
- TREND (CHOP < 45): Pullback entries to 4h HMA with RSI confirmation (RSI 30-50 long, 50-70 short)
- TRANSITION (45-55): No trades (avoid whipsaw)

Key innovations:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — faster than standard RSI
2. Choppiness Index regime filter — adapts strategy to market state
3. 4h HMA(21) bias — only long above, only short below (HTF alignment)
4. 12h HMA(50) meta-filter — stronger conviction when aligned with 4h
5. Session filter 08-20 UTC — avoid low liquidity periods
6. ATR(14) 2.5x trailing stop — risk management

Entry conditions (LOOSE to ensure trades):
- LONG range: CHOP > 55 AND CRSI < 25 AND close > 4h HMA
- LONG trend: CHOP < 45 AND RSI(14) < 50 AND close > 4h HMA AND close < 4h HMA * 1.02
- SHORT range: CHOP > 55 AND CRSI > 75 AND close < 4h HMA
- SHORT trend: CHOP < 45 AND RSI(14) > 50 AND close < 4h HMA AND close > 4h HMA * 0.98

Target: Sharpe > 0.40, trades >= 40/year (160+ train, 12+ test)
Timeframe: 1h
Size: 0.20 discrete (conservative to survive 2022 crash)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_hma_4h12h_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0.0)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    rs[:] = np.nan
    mask = avg_loss > 1e-10
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of past returns lower than current return
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period + 5, n):
        streak = 0
        if i > 0:
            if close[i] > close[i-1]:
                streak = 1
                for j in range(i-1, max(0, i-streak_period-5), -1):
                    if close[j] > close[j-1]:
                        streak += 1
                    else:
                        break
            elif close[i] < close[i-1]:
                streak = -1
                for j in range(i-1, max(0, i-streak_period-5), -1):
                    if close[j] < close[j-1]:
                        streak -= 1
                    else:
                        break
        
        # Calculate RSI of streak values
        if i >= streak_period:
            streak_values = np.zeros(streak_period + 1)
            valid_count = 0
            for k in range(streak_period + 1):
                idx = i - k
                if idx > 0:
                    if close[idx] > close[idx-1]:
                        streak_values[k] = 1
                    elif close[idx] < close[idx-1]:
                        streak_values[k] = -1
                    else:
                        streak_values[k] = 0
                    valid_count += 1
            
            if valid_count >= streak_period:
                gains = sum(1 for v in streak_values[:streak_period] if v > 0)
                losses = sum(1 for v in streak_values[:streak_period] if v < 0)
                if losses > 0:
                    streak_rsi[i] = 100.0 - (100.0 / (1.0 + gains / losses))
                else:
                    streak_rsi[i] = 100.0
    
    # Percent Rank
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0 and abs(returns[-1]) > 1e-10:
            count_lower = sum(1 for r in returns[:-1] if r < returns[-1])
            percent_rank[i] = 100.0 * count_lower / len(returns[:-1])
    
    # Combine
    for i in range(rank_period + 5, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index = 100 * log10(sum(ATR, period) / (highest_high - lowest_low)) / log10(period)
    
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    for i in range(period * 2, n):
        if np.isnan(atr[i]):
            continue
        
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return choppiness

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    rsi = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.20  # Conservative position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        hour = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour <= 20
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF CONFIRMATION (12h HMA) ===
        htf_strong_bull = htf_bull and close[i] > hma_12h_aligned[i]
        htf_strong_bear = htf_bear and close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        is_range = choppiness[i] > 55.0
        is_trend = choppiness[i] < 45.0
        # 45-55 = transition, no trades
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        if in_session:
            # RANGE REGIME: Mean reversion with CRSI
            if is_range:
                # Long: CRSI oversold + HTF bull
                if crsi[i] < 25.0 and htf_bull:
                    desired_signal = SIZE
                # Short: CRSI overbought + HTF bear
                elif crsi[i] > 75.0 and htf_bear:
                    desired_signal = -SIZE
            
            # TREND REGIME: Pullback entries
            elif is_trend:
                # Long: RSI pullback + HTF bull + near HMA
                if rsi[i] < 50.0 and htf_bull:
                    # Check if pulling back to HMA (within 2%)
                    hma_distance = (close[i] - hma_4h_aligned[i]) / hma_4h_aligned[i]
                    if hma_distance < 0.02:
                        desired_signal = SIZE
                    elif htf_strong_bull and rsi[i] < 45.0:
                        # Strong HTF, allow slightly further entry
                        desired_signal = SIZE * 0.5
                
                # Short: RSI pullback + HTF bear + near HMA
                elif rsi[i] > 50.0 and htf_bear:
                    hma_distance = (close[i] - hma_4h_aligned[i]) / hma_4h_aligned[i]
                    if hma_distance > -0.02:
                        desired_signal = -SIZE
                    elif htf_strong_bear and rsi[i] > 55.0:
                        desired_signal = -SIZE * 0.5
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < stop_price or low[i] < trailing_stop:
                stoploss_triggered = True
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > stop_price or high[i] > trailing_stop:
                stoploss_triggered = True
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.4:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.4:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals