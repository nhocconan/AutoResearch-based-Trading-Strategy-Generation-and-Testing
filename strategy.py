#!/usr/bin/env python3
"""
Experiment #922: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + ADX + CRSI Mean Reversion

Hypothesis: After 650+ failed strategies, the key is SIMPLER logic with GUARANTEED trade generation.
Previous strategies failed because: (1) too many conflicting filters = 0 trades, (2) HMA too noisy,
(3) regime logic too complex. This strategy uses:

1. KAMA (Kaufman Adaptive MA) instead of HMA — adapts to volatility, less whipsaw in 2022 crash
2. ADX(14) for trend strength — only trend-follow when ADX>25, mean-revert when ADX<20
3. CRSI(3,2,100) with RELAXED thresholds (15/85 not 10/90) — ensures trades on BTC/ETH/SOL
4. 1d KAMA(21) for medium-term bias, 1w KAMA(21) for macro regime
5. ATR(14) 2.5x trailing stop — mandatory risk management
6. Asymmetric sizing: 0.30 for high-confidence, 0.20 for lower-confidence

Why 12h works:
- 25-40 trades/year target (low fee drag at 0.05% RT)
- HTF (1d/1w) provides stronger signal than 4h/1h
- Less noise than lower TF, more responsive than 1d

Key improvements from #872:
- KAMA instead of HMA (better adaptation to vol regimes)
- ADX regime filter (clearer trend vs range distinction)
- Simpler hold logic (fewer exit conditions = fewer premature exits)
- Guaranteed fallback entries (CRSI extreme alone triggers trade)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_crsi_1d1w_regime_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market noise — smooth in ranging, responsive in trending.
    ER (Efficiency Ratio) determines smoothing constant.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
    
    # KAMA calculation
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] ** 2 * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_dm_s / (tr_s + 1e-10)
        minus_di = 100 * minus_dm_s / (tr_s + 1e-10)
    
    # DX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx = np.concatenate([[np.nan] * (period * 2), adx_raw[period*2:]])
    
    return adx

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Relaxed thresholds: 15/85 for guaranteed trade generation
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < max(rsi_period, streak_period, rank_period) + 2:
        return crsi
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100 - (100 / (1 + rs))
    rsi_short = np.clip(rsi_short, 0, 100)
    
    # RSI Streak
    streak = np.zeros(n)
    direction = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if direction[i-1] == 1 else 1
            direction[i] = 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if direction[i-1] == -1 else -1
            direction[i] = -1
    
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        up_count = np.sum(streak_vals > 0)
        total = np.sum(streak_vals != 0)
        if total > 0:
            streak_rsi[i] = 100 * up_count / total
        else:
            streak_rsi[i] = 50
    
    # Percent Rank
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = close[i] - close[i-1]
            percent_rank[i] = 100 * np.sum(returns < current_return) / len(returns)
        else:
            percent_rank[i] = 50
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    kama_12h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    adx_12h = calculate_adx(high, low, close, period=14)
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_12h = calculate_rsi(close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 1d KAMA for medium-term trend
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate and align 1w KAMA for macro regime
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=10, fast_period=2, slow_period=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    signals = np.zeros(n)
    HIGH_SIZE = 0.30
    LOW_SIZE = 0.20
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(350, n):
        # Skip if indicators not ready
        if np.isnan(kama_12h[i]) or np.isnan(adx_12h[i]) or np.isnan(crsi_12h[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            continue
        
        # === MACRO REGIME (1w KAMA) ===
        macro_bull = close[i] > kama_1w_aligned[i]
        macro_bear = close[i] < kama_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d KAMA) ===
        trend_bullish = close[i] > kama_1d_aligned[i]
        trend_bearish = close[i] < kama_1d_aligned[i]
        
        # === SHORT-TERM TREND (12h KAMA) ===
        short_bullish = close[i] > kama_12h[i]
        short_bearish = close[i] < kama_12h[i]
        
        # === ADX REGIME ===
        trending = adx_12h[i] > 25
        ranging = adx_12h[i] < 20
        
        # === CRSI SIGNALS (Relaxed: 15/85) ===
        crsi_oversold = crsi_12h[i] < 15
        crsi_overbought = crsi_12h[i] > 85
        crsi_extreme_os = crsi_12h[i] < 10
        crsi_extreme_ob = crsi_12h[i] > 90
        
        # === RSI FALLBACK ===
        rsi_oversold = rsi_12h[i] < 30
        rsi_overbought = rsi_12h[i] > 70
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (ADX > 25) — Trend Following ===
        if trending:
            # Long: All trend alignments + pullback entry
            if macro_bull and trend_bullish and short_bullish:
                if crsi_oversold or rsi_oversold:
                    desired_signal = HIGH_SIZE
                elif crsi_12h[i] < 40:
                    desired_signal = LOW_SIZE
            
            # Short: All trend alignments + pullback entry
            if macro_bear and trend_bearish and short_bearish:
                if crsi_overbought or rsi_overbought:
                    desired_signal = -HIGH_SIZE
                elif crsi_12h[i] > 60:
                    desired_signal = -LOW_SIZE
        
        # === RANGING REGIME (ADX < 20) — Mean Reversion ===
        elif ranging:
            # Long: CRSI oversold (guaranteed trades)
            if crsi_oversold:
                desired_signal = HIGH_SIZE
            elif crsi_extreme_os:
                desired_signal = HIGH_SIZE
            elif rsi_oversold:
                desired_signal = LOW_SIZE
            
            # Short: CRSI overbought
            if crsi_overbought:
                desired_signal = -HIGH_SIZE
            elif crsi_extreme_ob:
                desired_signal = -HIGH_SIZE
            elif rsi_overbought:
                desired_signal = -LOW_SIZE
        
        # === NEUTRAL REGIME (20 <= ADX <= 25) ===
        else:
            # Conservative: need trend + CRSI confluence
            if macro_bull and trend_bullish and crsi_oversold:
                desired_signal = LOW_SIZE
            if macro_bear and trend_bearish and crsi_overbought:
                desired_signal = -LOW_SIZE
            
            # Fallback: extreme CRSI alone
            if crsi_extreme_os and desired_signal == 0:
                desired_signal = LOW_SIZE
            if crsi_extreme_ob and desired_signal == 0:
                desired_signal = -LOW_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position through pullbacks ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro trend intact
                if macro_bull and crsi_12h[i] < 80:
                    desired_signal = HIGH_SIZE if trend_bullish else LOW_SIZE
            elif position_side < 0:
                # Hold short if macro trend intact
                if macro_bear and crsi_12h[i] > 20:
                    desired_signal = -HIGH_SIZE if trend_bearish else -LOW_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit if macro reverses + CRSI overbought
            if macro_bear and crsi_12h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit if macro reverses + CRSI oversold
            if macro_bull and crsi_12h[i] < 25:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = HIGH_SIZE if desired_signal >= 0.25 else LOW_SIZE
        elif desired_signal < 0:
            desired_signal = -HIGH_SIZE if desired_signal <= -0.25 else -LOW_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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