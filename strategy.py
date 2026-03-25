#!/usr/bin/env python3
"""
Experiment #1234: 1d Primary + 1w HTF — Connors RSI + Dual HMA Trend + Choppiness Regime

Hypothesis: Daily timeframe with weekly confirmation should produce 20-50 trades/year
with lower fee drag than lower TFs. Connors RSI (CRSI) has shown 0.923 Sharpe on ETH
in research, catching oversold/overbought extremes in trend direction.

Key components:
1. Weekly HMA(21) - Major trend bias (only trade in direction)
2. Daily HMA(21) - Intermediate trend confirmation
3. Connors RSI(3,2,100) - Entry timing at extremes (<30 long, >70 short)
4. Choppiness Index(14) - Regime filter (>61.8 = range, <38.2 = trend)
5. ATR(14) 2.5x trailing stop - Risk management

Entry logic (LOOSE to guarantee trades):
- LONG: 1w_HMA bullish + 1d_HMA bullish + CRSI < 40 (oversold in uptrend)
- SHORT: 1w_HMA bearish + 1d_HMA bearish + CRSI > 60 (overbought in downtrend)
- In choppy regime (CHOP > 61.8): widen CRSI thresholds to <25/>75 for mean reversion

Why this should work:
- 1d timeframe = natural 20-50 trades/year (fee-friendly)
- CRSI more responsive than standard RSI, catches pullbacks better
- Dual HMA (1w + 1d) = strong trend filter without being too restrictive
- Choppiness regime = adapt between trend-follow and mean-reversion
- Discrete sizing (0.0, ±0.25, ±0.30) = minimal fee churn

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_dual_hma_chop_regime_1w_v1"
timeframe = "1d"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Where current price ranks in last N days (0-100)
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - momentum of consecutive up/down days
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_sign = np.sign(streak)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(streak_period, n):
        streak_window = streak_abs[max(0, i-streak_period+1):i+1]
        if len(streak_window) >= streak_period:
            avg_streak = np.mean(streak_window)
            # Map streak to 0-100 scale (long streak = extreme)
            streak_rsi[i] = 50.0 + streak_sign[i] * min(50.0, avg_streak * 10.0)
    
    # Percent Rank - where current price ranks in last N days
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        if len(window) >= rank_period:
            count_below = np.sum(window[:-1] < close[i])
            percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = Choppy/Range-bound
    CHOP < 38.2 = Trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr_vals = calculate_atr(high, low, close, period)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(atr_vals[i]):
            atr_sum = np.nansum(atr_vals[i-period+1:i+1])
            highest = np.nanmax(high[i-period+1:i+1])
            lowest = np.nanmin(low[i-period+1:i+1])
            range_val = highest - lowest
            
            if range_val > 1e-10 and atr_sum > 0:
                chop[i] = 100.0 * np.log10(atr_sum / range_val) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    hma_1d = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
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
        
        if np.isnan(hma_1d[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = not np.isnan(chop[i]) and chop[i] > 61.8
        trending_regime = not np.isnan(chop[i]) and chop[i] < 38.2
        
        # === TREND DIRECTION (Weekly + Daily HMA) ===
        price_above_1d = close[i] > hma_1d[i]
        price_below_1d = close[i] < hma_1d[i]
        
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        crsi_val = crsi[i]
        
        if choppy_regime:
            # Mean reversion in choppy market
            # Long when CRSI very oversold, Short when CRSI very overbought
            if crsi_val < 25:
                desired_signal = SIZE_BASE  # Mean revert long
            elif crsi_val > 75:
                desired_signal = -SIZE_BASE  # Mean revert short
        else:
            # Trend following in trending market
            # LONG: Both HMA bullish + CRSI pullback (oversold in uptrend)
            if price_above_1d and price_above_1w:
                if crsi_val < 40:  # Oversold pullback in uptrend
                    desired_signal = SIZE_STRONG
            
            # SHORT: Both HMA bearish + CRSI pullback (overbought in downtrend)
            elif price_below_1d and price_below_1w:
                if crsi_val > 60:  # Overbought pullback in downtrend
                    desired_signal = -SIZE_STRONG
        
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