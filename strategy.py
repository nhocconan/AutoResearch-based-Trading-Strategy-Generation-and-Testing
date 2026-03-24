#!/usr/bin/env python3
"""
Experiment #154: 4h Primary + 12h/1d HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 153 experiments, the winning pattern for 4h is:
- Connors RSI (CRSI) has proven 75% win rate for mean reversion entries
- Choppiness Index with LITERATURE thresholds (61.8/38.2) for regime detection
- 12h HMA provides trend bias without being too slow for 4h entries
- Volume confirmation on breakouts reduces false signals
- Asymmetric sizing: 0.30 in trend regime, 0.20 in chop regime (risk management)
- Wider stoploss (3.0x ATR) for 4h timeframe swings

Key improvements over #142 (Sharpe=-0.150):
1. CRSI instead of simple RSI (3-component: RSI(3) + RSI_Streak(2) + PercentRank(100))
2. Proper CHOP thresholds from literature (61.8 for range, 38.2 for trend)
3. 12h HMA instead of 1d (better alignment with 4h entries)
4. Volume confirmation: taker_buy_volume > 1.2x average for breakout validation
5. Asymmetric position sizing based on regime confidence
6. Stoploss: 3.0x ATR (wider for 4h, reduces premature exits)
7. LOOSE entry filters to ensure >=30 trades/train, >=3/test on ALL symbols

Target: Sharpe>0.375 (beat #152), DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_hma_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - proven 75% win rate for mean reversion
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of prior closes lower than current close
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period):i+1]
        up_streaks = np.sum(streak_vals > 0)
        down_streaks = np.sum(streak_vals < 0)
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100.0 * up_streaks / total
        else:
            streak_rsi[i] = 50.0
    
    # Component 3: PercentRank - % of prior closes lower than current
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_lower = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_lower / rank_period
    
    # Combine components
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = np.zeros(n)
    vol_avg[:] = np.nan
    for i in range(period - 1, n):
        vol_avg[i] = np.mean(volume[i-period+1:i+1])
    
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values if "taker_buy_volume" in prices.columns else volume * 0.5
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=34)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30  # 30% in trending regime (higher confidence)
    SIZE_CHOP = 0.20   # 20% in choppy regime (lower confidence)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h HMA) ===
        htf_bull = close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index - LITERATURE THRESHOLDS) ===
        # CHOP > 61.8 = choppy/range (mean revert)
        # CHOP < 38.2 = trending (trend follow)
        # 38.2 - 61.8 = transition zone (use trend bias)
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        is_transition = not is_choppy and not is_trending
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / vol_avg[i] if vol_avg[i] > 1e-10 else 1.0
        vol_confirmed = vol_ratio > 1.2  # 20% above average
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        donchian_breakout_bull = close[i] > donchian_upper[i-1] and close[i-1] <= donchian_upper[i-1]
        donchian_breakout_bear = close[i] < donchian_lower[i-1] and close[i-1] >= donchian_lower[i-1]
        
        # === CONNORS RSI SIGNALS (proven 75% win rate) ===
        crsi_oversold = crsi[i] < 15.0  # extreme oversold
        crsi_overbought = crsi[i] > 85.0  # extreme overbought
        crsi_recover_long = crsi[i] > 30.0 and crsi[i-1] < 30.0  # crossing up through 30
        crsi_recover_short = crsi[i] < 70.0 and crsi[i-1] > 70.0  # crossing down through 70
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === DESIRED SIGNAL (Dual Regime Logic with CRSI) ===
        desired_signal = 0.0
        current_size = SIZE_TREND if is_trending else SIZE_CHOP
        
        if is_trending:
            # TREND REGIME: Follow breakouts with HTF bias + volume confirmation
            # LONG: breakout + HTF bull + volume + HMA bull
            if donchian_breakout_bull and htf_bull and vol_confirmed and hma_bull:
                desired_signal = current_size
            # SHORT: breakout + HTF bear + volume + HMA bear
            elif donchian_breakout_bear and htf_bear and vol_confirmed and hma_bear:
                desired_signal = -current_size
            # Fallback: CRSI recovery in trend direction
            elif crsi_recover_long and htf_bull and hma_bull:
                desired_signal = current_size * 0.7
            elif crsi_recover_short and htf_bear and hma_bear:
                desired_signal = -current_size * 0.7
        elif is_choppy:
            # CHOPPY REGIME: Mean revert with CRSI extremes
            # LONG: CRSI oversold + HTF not strongly bear
            if crsi_oversold and not htf_bear:
                desired_signal = current_size
            # SHORT: CRSI overbought + HTF not strongly bull
            elif crsi_overbought and not htf_bull:
                desired_signal = -current_size
            # Fallback: CRSI recovery
            elif crsi_recover_long and hma_bull:
                desired_signal = current_size * 0.7
            elif crsi_recover_short and hma_bear:
                desired_signal = -current_size * 0.7
        else:
            # TRANSITION ZONE: Use HTF bias + CRSI
            if crsi_oversold and htf_bull:
                desired_signal = SIZE_CHOP
            elif crsi_overbought and htf_bear:
                desired_signal = -SIZE_CHOP
            elif donchian_breakout_bull and htf_bull:
                desired_signal = SIZE_TREND * 0.7
            elif donchian_breakout_bear and htf_bear:
                desired_signal = -SIZE_TREND * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x for 4h) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_TREND * 0.85:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.85:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_CHOP * 0.85:
            final_signal = SIZE_CHOP
        elif desired_signal <= -SIZE_CHOP * 0.85:
            final_signal = -SIZE_CHOP
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
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals