#!/usr/bin/env python3
"""
Experiment #1530: 1h Primary + 4h HTF — Regime-Adaptive with Session Filter

Hypothesis: 1h can work IF we use HTF for direction and balance filters:
1. 4h HMA(21) provides macro trend bias (only trade in HTF direction)
2. 1h Choppiness Index(14) detects regime: CHOP>55=range, CHOP<45=trend
3. 1h Connors RSI for entry timing (extreme values for mean reversion)
4. Volume filter: volume > 0.7x 20-period average (avoid low liquidity)
5. Session filter: 8-20 UTC only (avoid Asian session whipsaws) - LOOSE
6. ATR 2.5x trailing stop for risk management
7. Position size 0.25 (smaller for 1h to reduce fee impact)

Key insight from failures:
- #1520, #1525: 1h strategies failed due to too many trades or wrong signals
- #1528, #1529: 30m/4h had 0 trades (filters too strict)
- Need BALANCE: loose enough for 40-80 trades/year, strict enough to avoid churn

Design:
- 4h HMA(21) aligned properly via mtf_data helper
- 1h Choppiness Index for regime detection
- 1h Connors RSI(3,2,100) for entry timing
- Volume > 0.7x avg 20
- Session 8-20 UTC only (but fallback without session for more trades)
- ATR(14) 2.5x trailing stop
- Signal size: 0.25 (discrete: 0.0, ±0.25)
- Target: 40-80 trades/train (4 years), 10-20 trades/test (15 months)

Timeframe: 1h (as required by experiment #1530)
HTF: 4h (trend bias)
Position Size: 0.25 (smaller for 1h vs 12h's 0.30)
Target: Sharpe > 0.618 (beat current best), DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_session_4h_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - combines RSI, streak, and percent rank"""
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        pos_streaks = np.sum(streak[i-streak_period+1:i+1] > 0)
        streak_rsi[i] = 100.0 * pos_streaks / streak_period if streak_period > 0 else 50.0
    
    # Percent Rank
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        current_return = returns[-1] if len(returns) > 0 else 0
        rank = np.sum(returns[:-1] < current_return) / max(len(returns) - 1, 1)
        percent_rank[i] = 100.0 * rank
    
    # Combine
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=16)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h (40-80 trades/year target)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1h[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # 1h bars: open_time is in milliseconds
        hour_utc = (prices["open_time"].iloc[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER (LOOSE) ===
        volume_ok = volume[i] > 0.7 * vol_avg[i]
        
        # === MACRO TREND (4h HMA) - primary direction bias ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        trend_regime = chop[i] < 45.0  # Trending market
        range_regime = chop[i] > 55.0  # Range/choppy market
        
        # === PRIMARY TREND (1h HMA) ===
        hma_bull = close[i] > hma_1h[i]
        hma_bear = close[i] < hma_1h[i]
        
        # === RSI CONDITIONS (LOOSE for more trades) ===
        rsi_oversold = rsi[i] < 45.0
        rsi_overbought = rsi[i] > 55.0
        
        # === CONNORS RSI (Mean Reversion Signal - LOOSE) ===
        crsi_oversold = not np.isnan(crsi[i]) and crsi[i] < 30.0
        crsi_overbought = not np.isnan(crsi[i]) and crsi[i] > 70.0
        
        # === DESIRED SIGNAL - REGIME ADAPTIVE ===
        desired_signal = 0.0
        
        # Confluence score (need at least 2 of: HTF trend, 1h trend, volume, session)
        confluence_long = 0
        confluence_short = 0
        
        if htf_bull:
            confluence_long += 2  # HTF trend is most important
        if htf_bear:
            confluence_short += 2
        if hma_bull:
            confluence_long += 1
        if hma_bear:
            confluence_short += 1
        if volume_ok:
            confluence_long += 1
            confluence_short += 1
        if in_session:
            confluence_long += 0.5
            confluence_short += 0.5
        
        # LONG SIGNALS
        if confluence_long >= 2.5:  # Need HTF bull + at least one more
            # Trend regime: follow trend
            if trend_regime and hma_bull and rsi_oversold:
                desired_signal = BASE_SIZE
            # Range regime: Connors RSI mean reversion
            elif range_regime and crsi_oversold:
                desired_signal = BASE_SIZE
            # Fallback: HTF bull + 1h HMA bull (ensures trades)
            elif htf_bull and hma_bull and rsi[i] < 55.0:
                desired_signal = BASE_SIZE * 0.6
            # Fallback 2: HTF bull + RSI oversold (simple, ensures trades)
            elif htf_bull and rsi_oversold:
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT SIGNALS
        elif confluence_short >= 2.5:  # Need HTF bear + at least one more
            # Trend regime: follow trend
            if trend_regime and hma_bear and rsi_overbought:
                desired_signal = -BASE_SIZE
            # Range regime: Connors RSI mean reversion
            elif range_regime and crsi_overbought:
                desired_signal = -BASE_SIZE
            # Fallback: HTF bear + 1h HMA bear (ensures trades)
            elif htf_bear and hma_bear and rsi[i] > 45.0:
                desired_signal = -BASE_SIZE * 0.6
            # Fallback 2: HTF bear + RSI overbought (simple, ensures trades)
            elif htf_bear and rsi_overbought:
                desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.8:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.8:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.6
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