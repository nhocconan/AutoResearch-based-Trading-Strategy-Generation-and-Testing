#!/usr/bin/env python3
"""
Experiment #1206: 12h Primary + 1d HTF — KAMA Adaptive Trend + Choppiness Regime + CRSI

Hypothesis: #1202 (KAMA+ADX+Chop) achieved Sharpe=0.450 on 12h. This version enhances:
1. KAMA (Kaufman Adaptive) instead of HMA — adapts ER to volatility, fewer whipsaws
2. Choppiness Index regime switch (61.8 chop / 38.2 trend) — proven meta-filter
3. Connors RSI for mean reversion in chop regime — 75% win rate in research
4. Donchian breakout + ADX filter for trend regime — confirms momentum
5. 1d KAMA for macro bias — only trade aligned with higher TF trend
6. Relaxed entry thresholds to ensure ≥30 trades/year (critical for passing)

Key improvements from #1202:
- Add CRSI for chop regime entries (more signals than pure BB)
- Add ADX > 20 filter for trend confirmation (reduce false breakouts)
- Use 1d KAMA instead of HMA (better adaptation to volatility changes)
- Ensure entry conditions not too strict (avoid 0-trade failure)

Target: 30-60 trades/year, Sharpe > 0.612 (beat current best)
Position Size: 0.28 discrete (conservative for 12h)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_crsi_donchian_adx_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_sc=2/11, slow_sc=2/31):
    """
    Kaufman Adaptive Moving Average — adapts to market efficiency.
    ER (Efficiency Ratio) = |net change| / sum of absolute changes
    SC (Smoothing Constant) = [ER * (fast - slow) + slow]^2
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + 1:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        net_change = abs(close[i] - close[i - er_period])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

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
    Connors RSI — composite mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    rsi_short = calculate_rsi(close, period=rsi_period)
    
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
        up_streaks = np.sum(streak[i-streak_period+1:i+1] > 0)
        streak_rsi[i] = (up_streaks / streak_period) * 100.0
    
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close) / np.where(close[:-1] > 1e-10, close[:-1], 1e-10)
    returns = np.concatenate([[0], returns])
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window <= current) / rank_period
        pct_rank[i] = rank * 100.0
    
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
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
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout indicator."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — trend strength indicator."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > low[i-1] - low[i] else 0
        minus_dm[i] = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1] else 0
    
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    dx = np.full(n, np.nan)
    for i in range(period * 2, n):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            sum_di = plus_di[i] + minus_di[i]
            if sum_di > 1e-10:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / sum_di
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for macro trend filter
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (12h) indicators
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    adx = calculate_adx(high, low, close, period=14)
    kama_12h = calculate_kama(close, er_period=10)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(adx[i]) or np.isnan(kama_12h[i]) or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d KAMA) ===
        macro_bull = close[i] > kama_1d_aligned[i]
        macro_bear = close[i] < kama_1d_aligned[i]
        
        # === PRIMARY TREND (12h KAMA) ===
        trend_bull = close[i] > kama_12h[i]
        trend_bear = close[i] < kama_12h[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Relaxed from 61.8 for more trades
        is_trending = chop[i] < 45.0  # Relaxed from 38.2 for more trades
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx[i] > 20.0  # Relaxed from 25 for more trades
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 25.0  # Relaxed from 20 for more trades
        crsi_overbought = crsi[i] > 75.0  # Relaxed from 80 for more trades
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === TRENDING REGIME: Donchian Breakout + ADX + Macro Alignment ===
        if is_trending and strong_trend:
            # Long: breakout + both KAMA bullish
            if breakout_long and macro_bull and trend_bull:
                desired_signal = BASE_SIZE
            # Short: breakout + both KAMA bearish
            elif breakout_short and macro_bear and trend_bear:
                desired_signal = -BASE_SIZE
        
        # === CHOPPY REGIME: Mean Reversion (CRSI) ===
        elif is_choppy:
            # Long: oversold + macro not strongly bearish
            if crsi_oversold and not macro_bear:
                desired_signal = BASE_SIZE
            # Short: overbought + macro not strongly bullish
            elif crsi_overbought and not macro_bull:
                desired_signal = -BASE_SIZE
        
        # === TRANSITION ZONE: KAMA crossover + CRSI confirmation ===
        else:
            # Long: price above both KAMA + CRSI not overbought
            if trend_bull and macro_bull and crsi[i] < 60.0:
                desired_signal = BASE_SIZE
            # Short: price below both KAMA + CRSI not oversold
            elif trend_bear and macro_bear and crsi[i] > 40.0:
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
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals