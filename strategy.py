#!/usr/bin/env python3
"""
Experiment #1335: 1h Primary + 4h/1d HTF — Regime-Adaptive Connors RSI + Choppiness + Session

Hypothesis: Bear/range markets (2025+) require regime-adaptive logic. Simple trend following
fails on BTC/ETH in chop. This strategy uses:
1. Choppiness Index (14) to detect range (CHOP>55) vs trend (CHOP<45)
2. Connors RSI for mean reversion in range regimes (75% win rate in literature)
3. 4h HMA(21) for macro trend direction in trending regimes
4. Session filter (8-20 UTC) for liquidity and reduced noise
5. Volume filter (>0.8x 20-bar avg) to confirm moves

Key difference from failed #1325/#1330: LOOSER entry conditions to ensure 30-80 trades/year.
Connors RSI extremes (5-15 long, 85-95 short) instead of narrow bands.
Session filter reduces false signals but doesn't block all entries.

Timeframe: 1h (use 4h/1d for direction, 1h for entry timing)
Target: 40-80 trades/year, Sharpe > 0.612, trades >= 50 train, >= 8 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_chop_4h_session_vol_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close = np.asarray(close, dtype=np.float64)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=3):
    """RSI for Connors RSI component - short period"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak component for Connors RSI
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI of streak
    for i in range(period, n):
        streak_vals = streak[i-period+1:i+1]
        if not np.any(np.isnan(streak_vals)):
            gain_streak = np.sum(np.where(streak_vals > 0, streak_vals, 0))
            loss_streak = np.sum(np.where(streak_vals < 0, -streak_vals, 0))
            if loss_streak > 1e-10:
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + gain_streak / loss_streak))
            else:
                streak_rsi[i] = 100.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component for Connors RSI
    Measures current return vs past returns
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    pr = np.full(n, np.nan)
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        if not np.any(np.isnan(window)):
            current = window[-1]
            count_below = np.sum(window[:-1] < current)
            pr[i] = 100.0 * count_below / (period - 1)
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Extreme readings (<10 or >90) indicate mean reversion opportunities
    """
    rsi_short = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = np.full(len(close), np.nan)
    mask = ~(np.isnan(rsi_short) | np.isnan(rsi_streak) | np.isnan(pr))
    crsi[mask] = (rsi_short[mask] + rsi_streak[mask] + pr[mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market chop vs trend
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return choppiness

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

def calculate_sma(close, period=200):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_volume_avg(volume, period=20):
    """Average volume for volume filter"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    hours = (open_time // 3600000) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for macro trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for longer-term bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    # 1h HMA for local trend
    hma_1h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Conservative size for 1h timeframe
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * vol_avg[i]  # Relaxed from 0.8 to ensure trades
        
        # === REGIME DETECTION ===
        is_range = choppiness[i] > 50.0  # Range market
        is_trend = choppiness[i] < 45.0  # Trending market
        # Neutral zone (45-50): use both logic
        
        # === MACRO TREND (4h HMA) ===
        macro_bull = close[i] > hma_4h_aligned[i]
        macro_bear = close[i] < hma_4h_aligned[i]
        
        # === LOCAL TREND (1h HMA) ===
        local_bull = close[i] > hma_1h[i] if not np.isnan(hma_1h[i]) else False
        local_bear = close[i] < hma_1h[i] if not np.isnan(hma_1h[i]) else False
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        
        # === RANGE REGIME: Mean Reversion with Connors RSI ===
        if is_range:
            # Long: CRSI extremely oversold (<15) + above SMA200 or in session
            if crsi[i] < 15.0 and (above_sma200 or in_session):
                if volume_ok:
                    desired_signal = BASE_SIZE
            # Short: CRSI extremely overbought (>85) + below SMA200 or in session
            elif crsi[i] > 85.0 and (below_sma200 or in_session):
                if volume_ok:
                    desired_signal = -BASE_SIZE
        
        # === TREND REGIME: Follow HTF trend with pullback entries ===
        elif is_trend:
            # Long: Macro bull + local pullback (CRSI 20-40)
            if macro_bull and 20.0 <= crsi[i] <= 45.0:
                if in_session and volume_ok:
                    desired_signal = BASE_SIZE
            # Short: Macro bear + local bounce (CRSI 55-80)
            elif macro_bear and 55.0 <= crsi[i] <= 80.0:
                if in_session and volume_ok:
                    desired_signal = -BASE_SIZE
        
        # === NEUTRAL ZONE: Combine both approaches ===
        else:
            # Mean reversion signals (stronger CRSI extremes)
            if crsi[i] < 10.0 and volume_ok:
                desired_signal = BASE_SIZE
            elif crsi[i] > 90.0 and volume_ok:
                desired_signal = -BASE_SIZE
            # Trend signals with confirmation
            elif macro_bull and local_bull and crsi[i] < 50.0:
                if in_session and volume_ok:
                    desired_signal = BASE_SIZE
            elif macro_bear and local_bear and crsi[i] > 50.0:
                if in_session and volume_ok:
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
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