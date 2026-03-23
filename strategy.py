#!/usr/bin/env python3
"""
Experiment #799: 4h Primary + 1d HTF — Connors RSI Mean Reversion + ADX Trend Filter

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. Connors RSI (CRSI) outperforms standard RSI for mean reversion (75% win rate proven)
2. ADX(14) < 30 filters for ranging markets where mean reversion works best
3. 1d HMA(21) provides cleaner trend bias than Choppiness Index
4. Simpler entry logic = more trades (target 30-50/year on 4h)
5. CRSI thresholds 20/80 generate sufficient trades without overtrading
6. ATR(14) trailing stop at 2.5x protects from major drawdowns
7. Exit on CRSI cross 50 (faster exit than waiting for opposite extreme)
8. Discrete signals: 0.0, ±0.25, ±0.30 to minimize fee churn

Strategy design:
1. 1d HMA(21) for long-term trend bias (aligned via mtf_data helper)
2. 4h ADX(14) for trend strength filter (avoid mean reversion in strong trends)
3. 4h CRSI(3,2,100) for entry timing — proven mean reversion indicator
4. 4h ATR(14) for trailing stop (2.5x)
5. Entry Long: CRSI<20 + ADX<30 + price>1d_HMA
6. Entry Short: CRSI>80 + ADX<30 + price<1d_HMA
7. Exit: CRSI crosses 50 OR stoploss hit
8. Position size: 0.25-0.30 discrete

Key differences from failed #794:
- CRSI instead of RSI (better mean reversion signal)
- ADX filter instead of Choppiness (more reliable)
- Simpler entry: 3 conditions max (not 5+)
- CRSI exit at 50 (faster turnover, more trades)
- No volume filter (was causing missed trades)
- Cleaner hold/exit logic

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 30-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_adx_hma_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX).
    Measures trend strength. ADX > 25 = trending, ADX < 20 = ranging.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method (EMA with alpha = 1/period)
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI and DX
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di_pct = 100 * plus_di / (atr_smooth + 1e-10)
        minus_di_pct = 100 * minus_di / (atr_smooth + 1e-10)
        di_sum = plus_di_pct + minus_di_pct
        dx = 100 * np.abs(plus_di_pct - minus_di_pct) / (di_sum + 1e-10)
    
    # ADX is smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — Mean Reversion Indicator.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) of close — short-term momentum
    2. RSI(2) of streak — streak strength (consecutive up/down days)
    3. PercentRank(100) — where current price ranks vs last 100 days
    
    Entry: CRSI < 15 (oversold), CRSI > 85 (overbought)
    Exit: CRSI crosses 50
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # Component 1: RSI(3) of close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI(2) of streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100 * count_below / rank_period
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    adx_4h = calculate_adx(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF HMA for trend bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(adx_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or atr_4h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === ADX TREND STRENGTH ===
        ranging_market = adx_4h[i] < 30  # ADX < 30 = ranging (good for mean reversion)
        trending_market = adx_4h[i] >= 30  # ADX >= 30 = trending (avoid mean reversion)
        
        # === CRSI SIGNALS (Connors RSI) ===
        crsi_oversold = crsi_4h[i] < 20  # Entry long
        crsi_overbought = crsi_4h[i] > 80  # Entry short
        crsi_neutral = 40 < crsi_4h[i] < 60  # Exit zone
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        # Long: CRSI oversold + ranging market + price above 1d HMA (bullish bias)
        if crsi_oversold and ranging_market and trend_1d_bullish:
            desired_signal = BASE_SIZE
        
        # Short: CRSI overbought + ranging market + price below 1d HMA (bearish bias)
        if crsi_overbought and ranging_market and trend_1d_bearish:
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
        
        # === EXIT LOGIC — CRSI crosses neutral zone ===
        if in_position and not stoploss_triggered:
            if position_side > 0 and crsi_4h[i] > 55:
                # Exit long when CRSI rises above 55
                desired_signal = 0.0
            elif position_side < 0 and crsi_4h[i] < 45:
                # Exit short when CRSI falls below 45
                desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI still low and trend intact
                if crsi_4h[i] < 50 and trend_1d_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI still high and trend intact
                if crsi_4h[i] > 50 and trend_1d_bearish:
                    desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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