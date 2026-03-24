#!/usr/bin/env python3
"""
Experiment #634: 1d Primary + 1w HTF — Dual Regime (Mean Revert + Trend Follow)

Hypothesis: Daily timeframe with regime detection adapts to market conditions better than pure trend.
- Choppiness > 61.8 = range market → Connors RSI mean reversion (75% win rate)
- Choppiness < 38.2 = trend market → HMA trend following
- 1w HMA provides macro bias filter to avoid counter-trend trades

Why this should work on 1d:
1. 1d timeframe = 20-50 trades/year, minimal fee drag
2. Regime detection prevents trend strategies in choppy 2025 bear/range market
3. Connors RSI extreme levels (10/90) catch reversals with high probability
4. LOOSE entry conditions ensure trades (avoid 0-trade failure from exp #625, #627, #629)
5. Conservative sizing (0.25-0.30) limits drawdown in 2022-style crashes

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_crsi_hma_1w_v1"
timeframe = "1d"
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
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) - short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        if streak[i] >= streak_period:
            streak_rsi[i] = 100.0
        elif streak[i] <= -streak_period:
            streak_rsi[i] = 0.0
        else:
            streak_rsi[i] = 50.0 + (streak[i] / streak_period) * 50.0
    
    # Percent Rank - where current return ranks vs last 100 days
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1]) / (close[i-rank_period:i] + 1e-10)
        current_return = returns[-1] if len(returns) > 0 else 0
        count_below = np.sum(returns[:-1] <= current_return)
        pct_rank[i] = (count_below / max(len(returns) - 1, 1)) * 100.0
    
    crsi = (rsi_short + streak_rsi + pct_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    hma_1d = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(hma_1d[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        is_choppy = chop[i] > 61.8  # Range market → mean reversion
        is_trending = chop[i] < 38.2  # Trend market → trend follow
        
        # === MACRO BIAS (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === LOCAL TREND (1d HMA) ===
        local_bull = close[i] > hma_1d[i]
        local_bear = close[i] < hma_1d[i]
        
        # === RSI PULLBACK ===
        rsi_ok_long = not np.isnan(rsi_14[i]) and rsi_14[i] < 60.0
        rsi_ok_short = not np.isnan(rsi_14[i]) and rsi_14[i] > 40.0
        
        desired_signal = 0.0
        
        # === MEAN REVERSION REGIME (Choppy market) ===
        if is_choppy:
            # Connors RSI mean reversion - LOOSE conditions to ensure trades
            # Long: CRSI < 25 (oversold) + not strongly bearish macro
            # Short: CRSI > 75 (overbought) + not strongly bullish macro
            
            if crsi[i] < 25.0 and not (macro_bear and local_bear):
                desired_signal = SIZE_BASE
            elif crsi[i] > 75.0 and not (macro_bull and local_bull):
                desired_signal = -SIZE_BASE
        
        # === TREND FOLLOWING REGIME (Trending market) ===
        elif is_trending:
            # HMA trend following with RSI pullback - LOOSE conditions
            # Long: price > HMA + RSI pullback + macro not bearish
            # Short: price < HMA + RSI bounce + macro not bullish
            
            if local_bull and rsi_ok_long and not macro_bear:
                if macro_bull:
                    desired_signal = SIZE_STRONG  # All align
                else:
                    desired_signal = SIZE_BASE  # Local trend only
            elif local_bear and rsi_ok_short and not macro_bull:
                if macro_bear:
                    desired_signal = -SIZE_STRONG  # All align
                else:
                    desired_signal = -SIZE_BASE  # Local trend only
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Reduced size, require stronger confirmation
            # Use CRSI extremes even in neutral regime
            if crsi[i] < 20.0 and local_bull:
                desired_signal = SIZE_BASE * 0.5
            elif crsi[i] > 80.0 and local_bear:
                desired_signal = -SIZE_BASE * 0.5
            # Fallback: simple trend with macro confirmation
            elif local_bull and macro_bull:
                desired_signal = SIZE_BASE * 0.5
            elif local_bear and macro_bear:
                desired_signal = -SIZE_BASE * 0.5
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
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