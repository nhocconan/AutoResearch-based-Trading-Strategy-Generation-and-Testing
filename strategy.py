#!/usr/bin/env python3
"""
Experiment #1582: 4h Primary + 1d/1w HTF — Fisher Transform + Connors RSI Reversal

Hypothesis: 4h timeframe balances trade frequency (30-60/year) with signal quality.
This strategy combines Ehlers Fisher Transform (reversal detection) with Connors RSI
(mean reversion) and 1d HMA trend filter. Works well in bear/range markets (2025+).

Key components:
1. 1d HMA(21) for major trend bias (avoid counter-trend trades)
2. 1w HMA(14) for ultra-long-term regime filter
3. 4h Fisher Transform(9) for reversal signals (crosses at -1.5/+1.5)
4. 4h Connors RSI(3,2,100) for mean reversion extremes
5. 4h ATR(14) for stoploss (2.5x ATR trailing)
6. Volume filter: only enter if volume > 0.8 * 20-bar avg (avoid low-liquidity traps)

Why this should work:
- Fisher Transform excels at catching reversals in bear/range markets (2022, 2025+)
- Connors RSI has 75% win rate on extreme readings
- 1d/1w HMA prevents major counter-trend disasters
- 4h TF = natural 30-60 trades/year (fee-efficient per Rule 10)
- LOOSE thresholds guarantee trades: Fisher > -1.5 (not -1.0), CRSI < 30 (not 20)

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: 1d_HMA bullish + Fisher crosses above -1.5 + CRSI < 35 + volume OK
- SHORT: 1d_HMA bearish + Fisher crosses below +1.5 + CRSI > 65 + volume OK

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_crsi_reversal_1d1w_v1"
timeframe = "4h"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to -1 to +1 range
    Highlights turning points with sharp peaks
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close := (high + low) / 2)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        price_range = highest - lowest
        
        if price_range < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else 0.0
        else:
            normalized = 2.0 * (close[i] - lowest) / price_range - 1.0
            normalized = max(-0.999, min(0.999, normalized))  # clamp to avoid log errors
            
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
            if i > 0 and not np.isnan(fisher[i-1]):
                fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i-1]  # smooth
        
        fisher_signal[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else fisher[i]
    
    return fisher, fisher_signal

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    Extreme readings (<10 or >90) indicate reversal opportunities
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # RSI on streak (consecutive up/down days)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan, dtype=np.float64)
    mask = avg_streak_loss != 0
    rs_streak = np.zeros(n)
    rs_streak[mask] = avg_streak_gain[mask] / avg_streak_loss[mask]
    rsi_streak[mask] = 100 - (100 / (1 + rs_streak[mask]))
    rsi_streak[avg_streak_loss == 0] = 100.0  # all gains
    
    # Percent Rank over 100 periods
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period - 1, n):
        window = close[i - rank_period + 1:i + 1]
        count_below = np.sum(window[:-1] < window[-1])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into Connors RSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    valid = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid] = (rsi_close[valid] + rsi_streak[valid] + percent_rank[valid]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=14)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume SMA for filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d and 1w HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 35
        crsi_overbought = crsi[i] > 65
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + Fisher cross up + CRSI oversold + volume OK
        if price_above_1d and fisher_cross_up and crsi_oversold and volume_ok:
            desired_signal = SIZE_STRONG
        
        # LONG (weaker): 1w bullish + Fisher cross up + CRSI very oversold
        elif price_above_1w and fisher_cross_up and crsi[i] < 25 and volume_ok:
            desired_signal = SIZE_BASE
        
        # SHORT: 1d bearish + Fisher cross down + CRSI overbought + volume OK
        elif price_below_1d and fisher_cross_down and crsi_overbought and volume_ok:
            desired_signal = -SIZE_STRONG
        
        # SHORT (weaker): 1w bearish + Fisher cross down + CRSI very overbought
        elif price_below_1w and fisher_cross_down and crsi[i] > 75 and volume_ok:
            desired_signal = -SIZE_BASE
        
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