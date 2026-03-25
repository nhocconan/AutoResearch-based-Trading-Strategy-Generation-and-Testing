#!/usr/bin/env python3
"""
Experiment #1436: 30m Primary + 4h/1d HTF — Regime-Adaptive cRSI + Choppiness

Hypothesis: 30m timeframe with strict confluence filters can achieve 40-80 trades/year
while maintaining positive Sharpe. This strategy combines:
1. 1d HMA(21) for major trend bias (avoid counter-trend in bear markets)
2. 4h Choppiness Index for regime detection (trend vs range)
3. 30m Connors RSI (cRSI) for entry timing with loose thresholds
4. Session filter: 08-20 UTC (avoid low liquidity Asian night hours)
5. ATR(14) trailing stoploss (signal→0 when stopped)
6. Discrete sizing: 0.0, ±0.20 (minimize fee churn)

Why this should work where others failed:
- cRSI is LOOSE (entry at <30 or >70, not <10 or >90) = guarantees trades
- Choppiness regime filter adapts to market conditions (2022 crash vs 2021 bull)
- 30m TF with 4h/1d HTF = natural 50-80 trades/year (not overtraded)
- Session filter removes 40% of bars = fewer false signals during low liquidity
- Simple logic = fewer conditions that can all fail simultaneously

Entry logic (LOOSE to guarantee trades):
- TREND REGIME (CHOP<50): Long if 1d_HMA bullish + cRSI<40, Short if 1d_HMA bearish + cRSI>60
- RANGE REGIME (CHOP>=50): Long if cRSI<25 + price<BB_lower, Short if cRSI>75 + price>BB_upper

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 30m
Size: 0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_regime_4h1d_v1"
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
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2)
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
        streak_vals = streak[max(0, i-streak_period):i+1]
        if len(streak_vals) >= streak_period:
            pos_count = np.sum(streak_vals > 0)
            streak_rsi[i] = 100 * pos_count / len(streak_vals)
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    returns = np.diff(close) / close[:-1]
    returns = np.insert(returns, 0, 0)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        if len(window) >= rank_period:
            count_below = np.sum(window < returns[i])
            percent_rank[i] = 100 * count_below / len(window)
    
    # Combine
    crsi = np.full(n, np.nan, dtype=np.float64)
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP = 100 * (ATR(1,14) / (Highest High(14) - Lowest Low(14))) * 100 / log10(14)
    
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            # ATR(1) = average of single-bar true ranges
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                tr_sum += tr
            atr_1 = tr_sum / period
            
            choppiness[i] = 100 * (atr_1 / (highest_high - lowest_low)) * 100 / np.log10(period)
    
    return choppiness

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower

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
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    chop_4h_raw = calculate_choppiness(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate 30m indicators
    hma_21 = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    signals = np.zeros(n)
    SIZE = 0.20
    
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
        
        if np.isnan(crsi[i]) or np.isnan(hma_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(chop_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === REGIME DETECTION (4h Choppiness) ===
        chop = chop_4h_aligned[i]
        is_trending = chop < 50  # midpoint between 38.2 and 61.8
        is_ranging = chop >= 50
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        if in_session:
            if is_trending:
                # TREND REGIME: follow 1d trend with cRSI pullback
                # LONG: 1d bullish + cRSI < 40 (pullback, not extreme)
                if price_above_1d and crsi[i] < 40:
                    desired_signal = SIZE
                
                # SHORT: 1d bearish + cRSI > 60 (pullback, not extreme)
                elif price_below_1d and crsi[i] > 60:
                    desired_signal = -SIZE
            
            else:
                # RANGE REGIME: mean revert at BB extremes
                # LONG: cRSI < 25 + price below BB lower
                if crsi[i] < 25 and close[i] < bb_lower[i]:
                    desired_signal = SIZE
                
                # SHORT: cRSI > 75 + price above BB upper
                elif crsi[i] > 75 and close[i] > bb_upper[i]:
                    desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
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