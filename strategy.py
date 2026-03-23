#!/usr/bin/env python3
"""
Experiment #257: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime

Hypothesis: After 200+ failed experiments with complex regime-switching, return to
proven 1d patterns that showed success in research:
- Connors RSI (CRSI) for entry timing: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Choppiness Index (CHOP) for regime detection: >61.8 = range (mean revert), <38.2 = trend
- 1w HMA(21) for macro bias (proven in best multi-TF strategies)
- ATR(14) 2.5x trailing stoploss

KEY INSIGHT FROM FAILURES:
- #247, #249, #252, #253, #254: Complex regime-switching created negative Sharpe
- #248, #250, #255: Too many filters = 0 trades (Sharpe=0.000)
- Research shows: CRSI + CHOP on 1d gave ETH Sharpe +0.923

TARGET: 20-50 trades/year on 1d, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
CRITICAL: CRSI thresholds 10/90 (not 15/85) to ensure sufficient trade frequency
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_crsi(close):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of close over last 100 days
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for short-term momentum
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI-Streak(2) - streak of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI of streak values (period=2)
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    streak_avg_gain = streak_gain.ewm(span=2, min_periods=2, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=2, min_periods=2, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # PercentRank(100) - percentile rank of close over last 100 days
    percent_rank = np.zeros(n)
    for i in range(100, n):
        window = close[i-99:i+1]  # 100-day window ending at i
        rank = np.sum(window[:-1] < window[-1])  # count how many prior closes are lower
        percent_rank[i] = rank / 99.0 * 100.0  # normalize to 0-100
    
    # Combine into CRSI
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    choppiness = np.zeros(n)
    
    for i in range(period, n):
        # Calculate ATR for each bar in the window
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and tr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Calculate 1w HMA for macro trend (aligned properly with shift(1))
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # ranging market - mean reversion
        is_trending = chop[i] < 38.2  # trending market - trend following
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 10 = oversold (long opportunity in range)
        # CRSI > 90 = overbought (short opportunity in range)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY in RANGING market: CRSI oversold + price above 1w HMA (bullish bias)
        if is_choppy and crsi_oversold and price_above_hma_1w:
            desired_signal = POSITION_SIZE_FULL
        
        # SHORT ENTRY in RANGING market: CRSI overbought + price below 1w HMA (bearish bias)
        elif is_choppy and crsi_overbought and price_below_hma_1w:
            desired_signal = -POSITION_SIZE_FULL
        
        # LONG ENTRY in TRENDING market: CRSI pullback + price above 1w HMA
        elif is_trending and crsi_oversold and price_above_hma_1w:
            desired_signal = POSITION_SIZE_FULL
        
        # SHORT ENTRY in TRENDING market: CRSI rally + price below 1w HMA
        elif is_trending and crsi_overbought and price_below_hma_1w:
            desired_signal = -POSITION_SIZE_FULL
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (take profit) ===
        # Exit long if CRSI becomes overbought (>80)
        if in_position and position_side > 0 and crsi[i] > 80.0:
            desired_signal = 0.0
        
        # Exit short if CRSI becomes oversold (<20)
        if in_position and position_side < 0 and crsi[i] < 20.0:
            desired_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes and position is against new regime logic
        if in_position and position_side > 0 and is_trending and price_below_hma_1w:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and is_trending and price_above_hma_1w:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if setup still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if still in valid regime
                if (is_choppy and price_above_hma_1w) or (is_trending and price_above_hma_1w and crsi[i] < 80):
                    desired_signal = POSITION_SIZE_HALF
            elif position_side < 0:
                # Hold short if still in valid regime
                if (is_choppy and price_below_hma_1w) or (is_trending and price_below_hma_1w and crsi[i] > 20):
                    desired_signal = -POSITION_SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals