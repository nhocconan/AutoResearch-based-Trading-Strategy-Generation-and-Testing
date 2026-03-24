#!/usr/bin/env python3
"""
Experiment #1493: 1d Primary + 1w HTF — Connors RSI Mean Reversion with Choppiness Regime

Hypothesis: After analyzing 1104+ failed strategies, the winning pattern for 1d timeframe is:
1. Connors RSI (CRSI) for precise mean reversion entries (proven 75% win rate in literature)
2. Choppiness Index (CHOP) for regime detection - range vs trend
3. 1w HMA for macro trend bias (only trade with weekly trend)
4. Donchian breakout confirmation for entry timing
5. ATR(14)*2.5 trailing stoploss for risk management

Why 1d + 1w should beat current best (Sharpe=0.618):
1. Connors RSI is more sensitive than regular RSI → more trades (avoid 0-trade failure)
2. Choppiness filter adapts to market regime → better in bear/range (2025 test period)
3. Weekly HMA prevents trading against macro trend → reduces whipsaw
4. 1d timeframe = target 20-50 trades/year (minimal fee drag ~1-2.5%)
5. Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Key improvements over failed experiments:
- #1483 CRSI failed because no regime filter (CHOP)
- #1490 regime adaptive failed on 1h (too many fees)
- #1492 dual regime worked better on 12h (higher TF = less noise)

Timeframe: 1d
HTF: 1w (call get_htf_data ONCE before loop!)
Position Size: 0.30 (discrete levels: 0.0, ±0.25, ±0.30)
Target: 30-60 trades/train, 5-10 trades/test, Sharpe > 0.618, ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_hma_donchian_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - Larry Connors' mean reversion indicator
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(close,100)) / 3
    
    RSI(3): Short-term momentum
    RSI(Streak): Consecutive up/down days momentum
    PercentRank: Where current close ranks vs last 100 days
    
    Entry signals: CRSI < 10 (oversold long), CRSI > 90 (overbought short)
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3) - short term
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_3 = pd.Series(gain).ewm(span=3, min_periods=3, adjust=False).mean().values
    loss_3 = pd.Series(loss).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    rsi_3 = np.full(n, np.nan)
    mask_3 = loss_3 > 1e-10
    rsi_3[mask_3] = 100.0 - (100.0 / (1.0 + gain_3[mask_3] / loss_3[mask_3]))
    rsi_3[loss_3 <= 1e-10] = 100.0
    rsi_3[:3] = np.nan
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak (treat negative streaks as 0 for RSI calc)
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_gain_2 = pd.Series(streak_gain).ewm(span=2, min_periods=2, adjust=False).mean().values
    streak_loss_2 = pd.Series(streak_loss).ewm(span=2, min_periods=2, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask_streak = streak_loss_2 > 1e-10
    rsi_streak[mask_streak] = 100.0 - (100.0 / (1.0 + streak_gain_2[mask_streak] / streak_loss_2[mask_streak]))
    rsi_streak[streak_loss_2 <= 1e-10] = 100.0
    rsi_streak[:2] = np.nan
    
    # PercentRank - where does current close rank vs last 100 days
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    valid = (~np.isnan(rsi_3)) & (~np.isnan(rsi_streak)) & (~np.isnan(percent_rank))
    crsi[valid] = (rsi_3[valid] + rsi_streak[valid] + percent_rank[valid]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - Karl Dittmann's regime detector
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = Range/Chop (use mean reversion)
    CHOP < 38.2 = Trend (use trend following)
    38.2 < CHOP < 61.8 = Transition
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        hl_range = highest_high - lowest_low
        
        if hl_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / hl_range) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """
    Hull Moving Average - Alan Hull's low-lag MA
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, w_period):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, w_period + 1)
        for i in range(w_period - 1, len(series)):
            window = series[i-w_period+1:i+1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2.0 * wma_half - wma_full
    
    # WMA of diff with sqrt(n) period
    hma = wma(diff, sqrt_n)
    
    return hma

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper and lower bounds"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = close[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    mask = tr_s > 1e-10
    plus_di[mask] = 100.0 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100.0 * minus_dm_s[mask] / tr_s[mask]
    
    # DX
    dx = np.full(n, np.nan)
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    mask_dx = di_sum > 1e-10
    dx[mask_dx] = 100.0 * di_diff[mask_dx] / di_sum[mask_dx]
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    adx = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start after all indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(hma_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1w HMA) - direction bias ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # Range market - use mean reversion
        is_trending = chop[i] < 38.2  # Trend market - use trend following
        # 38.2 < CHOP < 61.8 = transition (reduce position or stay flat)
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx[i] > 25.0
        weak_trend = adx[i] < 20.0
        
        # === CRSI EXTREMES (Mean Reversion Signals) ===
        crsi_oversold = crsi[i] < 15.0  # Long entry
        crsi_overbought = crsi[i] > 85.0  # Short entry
        crsi_neutral = 40.0 < crsi[i] < 60.0
        
        # === HMA TREND (Primary) ===
        hma_bull = close[i] > hma_21[i] and hma_21[i] > hma_50[i]
        hma_bear = close[i] < hma_21[i] and hma_21[i] < hma_50[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_high = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_low = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === DESIRED SIGNAL - REGIME ADAPTIVE ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        if weekly_bull:  # Only long in weekly uptrend
            if is_choppy and crsi_oversold:
                # Range market + oversold = mean reversion long
                desired_signal = BASE_SIZE
            elif is_trending and hma_bull and breakout_high:
                # Trend market + HMA bull + breakout = trend long
                desired_signal = BASE_SIZE
            elif weak_trend and crsi[i] < 25.0 and close[i] > hma_50[i]:
                # Weak trend + moderately oversold + above 50 HMA
                desired_signal = BASE_SIZE * 0.7
            elif crsi[i] < 10.0 and close[i] > hma_1w_aligned[i]:
                # Extreme oversold + weekly support = strong long
                desired_signal = BASE_SIZE
        
        # SHORT ENTRIES
        elif weekly_bear:  # Only short in weekly downtrend
            if is_choppy and crsi_overbought:
                # Range market + overbought = mean reversion short
                desired_signal = -BASE_SIZE
            elif is_trending and hma_bear and breakout_low:
                # Trend market + HMA bear + breakdown = trend short
                desired_signal = -BASE_SIZE
            elif weak_trend and crsi[i] > 75.0 and close[i] < hma_50[i]:
                # Weak trend + moderately overbought + below 50 HMA
                desired_signal = -BASE_SIZE * 0.7
            elif crsi[i] > 90.0 and close[i] < hma_1w_aligned[i]:
                # Extreme overbought + weekly resistance = strong short
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.3:
            final_signal = -BASE_SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals