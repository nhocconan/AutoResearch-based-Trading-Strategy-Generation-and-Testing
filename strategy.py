#!/usr/bin/env python3
"""
Experiment #1177: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: After 860+ failed experiments, clear patterns emerge for 1d timeframe:
- 1d is proven to work best for swing trading (20-50 trades/year target)
- Choppiness Index regime detection switches between mean-revert (chop) and trend-follow
- Connors RSI (CRSI) has 75% win rate for mean reversion in range markets
- 1w HMA provides macro trend filter without being too slow
- FEWER filters = more trades (avoid 0-trade failures like #1165, #1166, #1168)
- ATR 2.5x trailing stop appropriate for 1d volatility
- Position size 0.30 discrete balances returns vs drawdown

Why this should beat Sharpe=0.612:
- Regime-switching adapts to bear/range markets (2022 crash, 2025 bear)
- Connors RSI catches oversold/overbought extremes with high win rate
- 1w HMA filter prevents counter-trend trades in major moves
- Simpler logic = more trades on ALL symbols (BTC, ETH, SOL must have Sharpe>0)
- Target: 25-45 trades/year on 1d, Sharpe > 0.612

Timeframe: 1d (primary)
HTF: 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.30 base (discrete: 0.0, ±0.30)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_crsi_hma_1w_atr_v1"
timeframe = "1d"
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
    """
    Relative Strength Index — momentum oscillator.
    """
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
    Connors RSI — composite mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    CRSI < 10 = oversold (long), CRSI > 90 = overbought (short)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak(2) — streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    abs_streak = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if abs_streak[i] == 0:
            streak_rsi[i] = 50.0
        else:
            # Map streak magnitude to 0-100 scale
            streak_rsi[i] = 100.0 / (1.0 + abs_streak[i])
            if streak[i] < 0:
                streak_rsi[i] = 100.0 - streak_rsi[i]
    
    # Percent Rank(100)
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period - 1, n):
        window = close[i - rank_period + 1:i + 1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        pct_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    for i in range(rank_period - 1, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
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
    
    for i in range(period - 1, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
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
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    atr = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    hma_1d = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    
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
        if np.isnan(atr[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d[i]) or np.isnan(sma_200[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === LOCAL TREND (1d HMA) ===
        local_bull = close[i] > hma_1d[i]
        local_bear = close[i] < hma_1d[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Range market (mean revert)
        is_trending = chop[i] < 45.0  # Trend market (trend follow)
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Strong long signal
        crsi_overbought = crsi[i] > 85.0  # Strong short signal
        crsi_moderate_oversold = crsi[i] < 25.0  # Moderate long
        crsi_moderate_overbought = crsi[i] > 75.0  # Moderate short
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === LONG ENTRY — Mean Reversion in Choppy Market ===
        # CRSI oversold + choppy regime + above SMA200 + macro bull bias
        if is_choppy and crsi_oversold and above_sma200:
            desired_signal = BASE_SIZE
        # CRSI moderately oversold + trending + all trend filters align
        elif is_trending and crsi_moderate_oversold and macro_bull and local_bull and above_sma200:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY — Mean Reversion in Choppy Market ===
        # CRSI overbought + choppy regime + below SMA200 + macro bear bias
        elif is_choppy and crsi_overbought and below_sma200:
            desired_signal = -BASE_SIZE
        # CRSI moderately overbought + trending + all trend filters align
        elif is_trending and crsi_moderate_overbought and macro_bear and local_bear and below_sma200:
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
                # Hold long if CRSI not overbought and trend intact
                if crsi[i] < 80.0 and macro_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI not oversold and trend intact
                if crsi[i] > 20.0 and macro_bear:
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