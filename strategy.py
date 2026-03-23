#!/usr/bin/env python3
"""
Experiment #1166: 12h Primary + 1d HTF — Dual Regime (Choppiness + Connors RSI + Donchian)

Hypothesis: After analyzing 852+ failed experiments, clear patterns emerge:
- 12h timeframe needs REGIME DETECTION to work (simple trend fails: #1162 Sharpe=-0.724)
- Choppiness Index > 61.8 = range market → use mean reversion (Connors RSI)
- Choppiness Index < 38.2 = trend market → use trend follow (Donchian breakout)
- Connors RSI (RSI3 + RSI_Streak2 + PercentRank100)/3 triggers faster than RSI14
- 1d HTF filter prevents counter-macro trades
- Looser entry thresholds ensure 30+ trades (avoid 0-trade failures like #1155, #1157, #1159)

Why this should beat Sharpe=0.612:
- Dual regime adapts to market conditions (bull/bear/range)
- Connors RSI proven on ETH with Sharpe +0.923 in research
- Choppiness filter prevents trend strategies in chop (major failure mode)
- 12h naturally produces 20-50 trades/year (optimal fee drag)
- Position size 0.30 discrete with 2.5x ATR stoploss

Timeframe: 12h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.30 base (discrete: 0.0, ±0.30)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_crsi_donchian_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        range_hl = highest - lowest
        
        if range_hl > 1e-10 and sum_tr > 0:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — combines 3 components for mean reversion signals.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    CRSI < 10 = oversold (long), CRSI > 90 = overbought (short)
    For entries: CRSI < 20 long, CRSI > 80 short (looser for more trades)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.zeros(n)
    mask = loss_smooth > 1e-10
    rsi_close[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi_close[~mask] = 100.0
    
    # Component 2: RSI of streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    streak_gain = np.insert(streak_gain, 0, 0)
    streak_loss = np.insert(streak_loss, 0, 0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    mask2 = streak_loss_smooth > 1e-10
    rsi_streak[mask2] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[mask2] / streak_loss_smooth[mask2]))
    rsi_streak[~mask2] = 100.0
    
    # Component 3: Percent Rank of recent returns
    percent_rank = np.full(n, np.nan)
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / close[:-1] * 100.0
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current)
        percent_rank[i] = rank / rank_period * 100.0
    
    # Combine
    valid = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid] = (rsi_close[valid] + rsi_streak[valid] + percent_rank[valid]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """
    Donchian Channels — breakout detection.
    Upper = Highest High(period), Lower = Lowest Low(period)
    Breakout above upper = long signal, below lower = short signal
    """
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

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

def calculate_hma(close, period=21):
    """Hull Moving Average for trend direction."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_12h = calculate_hma(close, period=21)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_12h[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === LOCAL TREND (12h HMA) ===
        local_bull = close[i] > hma_12h[i]
        local_bear = close[i] < hma_12h[i]
        
        # === REGIME DETECTION (Choppiness) ===
        # CHOP > 61.8 = range/chop (mean revert)
        # CHOP < 38.2 = trending (trend follow)
        # 38.2-61.8 = transition (no trade or reduced size)
        is_choppy = chop[i] > 55.0  # Slightly lower threshold for more signals
        is_trending = chop[i] < 45.0  # Slightly higher threshold for more signals
        
        # === MEAN REVERSION SIGNALS (Connors RSI in choppy regime) ===
        # CRSI < 20 = oversold (long), CRSI > 80 = overbought (short)
        crsi_oversold = crsi[i] < 25.0  # Looser for more trades
        crsi_overbought = crsi[i] > 75.0  # Looser for more trades
        
        # === TREND FOLLOWING SIGNALS (Donchian breakout in trending regime) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Regime 1: Choppy + CRSI oversold + macro not strongly bear
        if is_choppy and crsi_oversold and not macro_bear:
            desired_signal = BASE_SIZE
        
        # Regime 2: Trending + Donchian breakout + macro bull + local bull
        elif is_trending and donchian_breakout_long and macro_bull and local_bull:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Regime 1: Choppy + CRSI overbought + macro not strongly bull
        if desired_signal == 0.0 and is_choppy and crsi_overbought and not macro_bull:
            desired_signal = -BASE_SIZE
        
        # Regime 2: Trending + Donchian breakout + macro bear + local bear
        elif desired_signal == 0.0 and is_trending and donchian_breakout_short and macro_bear and local_bear:
            desired_signal = -BASE_SIZE
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull:
            desired_signal = 0.0
        
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if conditions still valid
                hold_long_chop = is_choppy and crsi[i] < 50.0 and not macro_bear
                hold_long_trend = is_trending and macro_bull and local_bull
                if hold_long_chop or hold_long_trend:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if conditions still valid
                hold_short_chop = is_choppy and crsi[i] > 50.0 and not macro_bull
                hold_short_trend = is_trending and macro_bear and local_bear
                if hold_short_chop or hold_short_trend:
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