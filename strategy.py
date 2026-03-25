#!/usr/bin/env python3
"""
Experiment #1356: 30m Primary + 4h/1d HTF — CRSI Mean Reversion + HTF Trend + Session Filter

Hypothesis: Connors RSI (CRSI) provides superior mean-reversion signals compared to standard RSI,
especially in bear/range markets (2025 test period). Combined with 4h HMA for trend direction
and session filter (08-20 UTC) for liquidity, this should achieve:
- 40-80 trades/year on 30m timeframe (fee-friendly)
- Positive Sharpe on BTC/ETH/SOL individually
- DD < -35% via conservative sizing (0.20-0.30)

Key features:
1. CRSI(3,2,100) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — proven 75% win rate
2. 4h HMA(21) for major trend bias (only long when price > 4h HMA)
3. 1d HMA(21) for regime filter (avoid counter-trend in strong regimes)
4. Session filter: 08-20 UTC only (avoid low-liquidity Asian overnight)
5. ATR(14) 2.5x trailing stop
6. Discrete sizing (0.0, ±0.20, ±0.30)

Why this should work where others failed:
- CRSI is more sensitive than RSI(14) — catches reversals faster
- 4h trend filter prevents counter-trend deaths in 2022 crash
- Session filter reduces false breakouts during low-volume periods
- 30m TF = natural 40-80 trades/year (fee-friendly, not too many)
- Conservative sizing (0.20-0.30) limits drawdown during crashes

Entry logic:
- LONG: 4h price > 4h_HMA + 30m CRSI < 15 + session 08-20 UTC
- SHORT: 4h price < 4h_HMA + 30m CRSI > 85 + session 08-20 UTC

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 30m
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_mean_reversion_hma_trend_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
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
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) - fast RSI for sensitivity
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI Streak - measures consecutive up/down days
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        streak_window = streak[max(0, i-streak_period+1):i+1]
        if len(streak_window) >= streak_period:
            up_streaks = np.sum(streak_window > 0)
            down_streaks = np.sum(streak_window < 0)
            total = up_streaks + down_streaks
            if total > 0:
                streak_rsi[i] = 100 * up_streaks / total
            else:
                streak_rsi[i] = 50
    
    # Percent Rank - where does today's return rank vs last 100 days?
    returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i-1] != 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100
    
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            current = returns[i]
            rank = np.sum(valid < current) / len(valid) * 100
            percent_rank[i] = rank
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_fast[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = range/consolidation
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
        else:
            atr_sum = 0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def get_session_hour(open_time_ms):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    return (open_time_ms // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only) ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten or hold, don't enter new
            if in_position:
                signals[i] = signals[i-1] if i > 0 else 0.0
            else:
                signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # 1d HMA for major regime (avoid strong counter-trend)
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === CRSI MEAN REVERSION ===
        crsi_val = crsi[i]
        crsi_oversold = crsi_val < 15  # Very oversold
        crsi_overbought = crsi_val > 85  # Very overbought
        
        # === CHOPPINESS FILTER ===
        chop_val = chop[i] if not np.isnan(chop[i]) else 50
        is_ranging = chop_val > 55  # Prefer mean reversion in ranges
        is_trending = chop_val < 45  # Can trend-follow
        
        # === ENTRY LOGIC (CRSI + HTF Trend + Session) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + CRSI oversold + session + preferably ranging
        if price_above_4h and crsi_oversold:
            if price_above_1d:
                # Strong alignment: 4h and 1d both bullish
                desired_signal = SIZE_STRONG
            elif is_ranging:
                # Ranging market: mean reversion long
                desired_signal = SIZE_BASE
            # Skip if 1d bearish and trending (counter-trend risk)
        
        # SHORT: 4h bearish + CRSI overbought + session + preferably ranging
        elif price_below_4h and crsi_overbought:
            if price_below_1d:
                # Strong alignment: 4h and 1d both bearish
                desired_signal = -SIZE_STRONG
            elif is_ranging:
                # Ranging market: mean reversion short
                desired_signal = -SIZE_BASE
            # Skip if 1d bullish and trending (counter-trend risk)
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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