#!/usr/bin/env python3
"""
Experiment #1170: 1h Primary + 4h/12h HTF — Regime-Adaptive CRSI + HMA Trend

Hypothesis: After 856+ failed experiments, the key insight is:
- 1h timeframe needs REGIME detection to avoid whipsaws (Choppiness Index)
- Connors RSI (CRSI) has 75% win rate for mean reversion in range markets
- 4h HMA provides trend direction filter (not too strict like 1d)
- 12h HMA provides macro bias (only trade with macro trend)
- Session filter (8-20 UTC) captures high-volume periods
- Volume filter ensures liquidity (avoid low-volume traps)

Why this should beat Sharpe=0.612:
- Regime-adaptive: mean revert in chop, trend follow in trends
- CRSI more responsive than standard RSI for pullback entries
- 4h HTF less laggy than 1d for 1h entries
- Session + volume filters reduce false signals without killing trade count
- Target: 40-80 trades/year on 1h, Sharpe > 0.612

Timeframe: 1h (primary)
HTF: 4h (trend), 12h (macro bias) — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 (smaller for 1h to reduce fee impact)
Stoploss: 2.0x ATR trailing (tighter for 1h)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_hma_4h12h_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
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
    """Relative Strength Index — momentum oscillator."""
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — composite momentum indicator for mean reversion.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    CRSI < 10 = extreme oversold (long), CRSI > 90 = extreme overbought (short)
    For pullback entries: CRSI < 20 (long), CRSI > 80 (short)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) — streak of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        avg_streak = np.mean(streak_vals)
        # Map streak to 0-100 range
        streak_rsi[i] = 50.0 + avg_streak * 10.0
        streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    # Percent Rank (100) — where current return ranks vs last 100 periods
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0], returns])
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / len(window)
        pct_rank[i] = rank * 100.0
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = range/choppy (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        sum_atr = np.sum(atr_smooth[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        range_hl = highest_high - lowest_low
        if range_hl > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_atr / range_hl) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
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

def calculate_sma(close, period=200):
    """Simple Moving Average for trend filter."""
    n = len(close)
    sma = np.full(n, np.nan)
    
    if n < period:
        return sma
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMAs for trend filters
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_1h = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    
    # Volume SMA for volume filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h to reduce fee impact
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(hma_1h[i]) or np.isnan(sma_200[i]) or np.isnan(vol_sma[i]):
            continue
        if atr[i] <= 1e-10 or vol_sma[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] >= 0.8 * vol_sma[i]
        
        # === MACRO TREND (12h HMA) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === LOCAL TREND (1h HMA + SMA200) ===
        local_bull = close[i] > hma_1h[i] and close[i] > sma_200[i]
        local_bear = close[i] < hma_1h[i] and close[i] < sma_200[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range/choppy (use mean reversion)
        # CHOP < 45 = trending (use trend following)
        is_range = chop[i] > 55.0
        is_trend = chop[i] < 45.0
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === RANGE REGIME: Mean Reversion with CRSI ===
        if is_range:
            # Long: CRSI < 20 (oversold pullback) + macro/4h bull bias
            if crsi[i] < 20.0 and macro_bull and trend_4h_bull and in_session and volume_ok:
                desired_signal = BASE_SIZE
            
            # Short: CRSI > 80 (overbought pullback) + macro/4h bear bias
            elif crsi[i] > 80.0 and macro_bear and trend_4h_bear and in_session and volume_ok:
                desired_signal = -BASE_SIZE
        
        # === TREND REGIME: Trend Following with Pullback ===
        elif is_trend:
            # Long: pullback to HMA in uptrend + CRSI not overbought
            if trend_4h_bull and macro_bull and close[i] <= hma_1h[i] * 1.005:
                if crsi[i] < 60.0 and in_session and volume_ok:
                    desired_signal = BASE_SIZE
            
            # Short: pullback to HMA in downtrend + CRSI not oversold
            elif trend_4h_bear and macro_bear and close[i] >= hma_1h[i] * 0.995:
                if crsi[i] > 40.0 and in_session and volume_ok:
                    desired_signal = -BASE_SIZE
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull:
            desired_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # If regime flips from trend to range (or vice versa), reduce exposure
        if in_position:
            if position_side > 0 and chop[i] > 61.8 and crsi[i] > 50.0:
                desired_signal = 0.0  # Range regime + CRSI neutral = exit long
            elif position_side < 0 and chop[i] > 61.8 and crsi[i] < 50.0:
                desired_signal = 0.0  # Range regime + CRSI neutral = exit short
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still intact
                if (is_range and crsi[i] < 50.0) or (is_trend and trend_4h_bull and macro_bull):
                    if in_session or True:  # Allow holding outside session
                        desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend still intact
                if (is_range and crsi[i] > 50.0) or (is_trend and trend_4h_bear and macro_bear):
                    if in_session or True:
                        desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
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