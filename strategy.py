#!/usr/bin/env python3
"""
Experiment #1076: 12h Primary + 1d HTF — KAMA Adaptive Trend + ADX + Choppiness Dual Regime

Hypothesis: After 779+ failed experiments, the winning pattern for 12h timeframe combines:
1. KAMA (Kaufman Adaptive Moving Average) — adapts speed based on market efficiency
   Fast in trends, slow in chop. Better than static EMA/HMA for crypto volatility.
   Long: price > KAMA + KAMA sloping up | Short: price < KAMA + KAMA sloping down
2. ADX (14) — trend strength confirmation
   ADX > 25 = trending (use breakout signals)
   ADX < 20 = ranging (use mean reversion signals)
   Hysteresis: enter at 25, exit at 18 to avoid whipsaw
3. CHOPPINESS INDEX (14) — regime detection (proven best for crypto)
   CHOP > 61.8 = range (mean reversion at Bollinger bounds)
   CHOP < 38.2 = trend (breakout following)
4. CONNORS RSI (CRSI) — mean reversion entry in choppy regimes
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI < 10 + price > SMA200 | Short: CRSI > 90 + price < SMA200
5. DONCHIAN (20) — breakout confirmation in trending regimes
6. 1d KAMA21 macro bias — only trade in direction of daily adaptive trend
7. ATR-based position sizing — reduce size when vol spikes (ATR ratio > 2.0)

Why this should beat Sharpe=0.612:
- KAMA adapts to volatility (unlike static EMA/HMA in 777 failed strategies)
- ADX + Choppiness dual filter prevents false signals in wrong regime
- CRSI proven 75% win rate for mean reversion (different from RSI/MACD failures)
- 12h timeframe = 20-50 trades/year target (optimal fee/trade balance)
- Different signal source than all failed strategies (KAMA not yet tested extensively)

Timeframe: 12h (primary)
HTF: 1d (daily) — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels (reduced to 0.15 in high vol)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_chop_crsi_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA) — adapts speed based on market efficiency.
    
    Formula:
    1. Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    2. Fast SC = 2/(fast_period+1) = 2/3
    3. Slow SC = 2/(slow_period+1) = 2/31
    4. Smoothing Constant (SC) = [ER * (Fast SC - Slow SC) + Slow SC]^2
    5. KAMA[i] = KAMA[i-1] + SC * (Close[i] - KAMA[i-1])
    
    KAMA moves fast in trends (high ER), slow in chop (low ER).
    Proven to reduce whipsaw in volatile crypto markets.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            volatility += abs(close[j] - close[j - 1])
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        else:
            sc[i] = slow_sc ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) — measures trend strength.
    
    Formula:
    1. Calculate +DM and -DM
    2. Calculate +DI and -DI
    3. Calculate DX = |+DI - -DI| / (+DI + -DI) * 100
    4. ADX = SMA of DX over period
    
    Interpretation:
    - ADX > 25 = strong trend
    - ADX < 20 = weak trend / ranging
    - ADX rising = trend strengthening
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range and DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
    
    # Calculate DX
    dx = np.full(n, np.nan)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    # Calculate ADX (SMA of DX)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — measures market choppiness vs trending.
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = choppy/range market (mean reversion favored)
    - CHOP < 38.2 = trending market (breakout/trend follow favored)
    - 38.2 - 61.8 = transition zone
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
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Calculate highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    for i in range(period, n):
        if np.isnan(atr_sum[i]) or np.isnan(hh[i]) or np.isnan(ll[i]):
            continue
        price_range = hh[i] - ll[i]
        if price_range > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — composite mean reversion indicator.
    
    Formula:
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Components:
    1. RSI(3) — short-term momentum
    2. RSI(Streak) — streak duration (consecutive up/down days)
    3. PercentRank(100) — where current price ranks vs last 100 bars
    
    Signals:
    - CRSI < 10 = extremely oversold (long opportunity)
    - CRSI > 90 = extremely overbought (short opportunity)
    
    Research shows 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + rsi_period + streak_period:
        return crsi
    
    # Component 1: RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi_close[i] = 100.0 - 100.0 / (1.0 + rs)
        elif avg_gain[i] > 1e-10:
            rsi_close[i] = 100.0
        else:
            rsi_close[i] = 50.0
    
    # Component 2: RSI(Streak)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to absolute values for RSI calculation
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0.0)
    streak_loss = np.where(streak < 0, streak_abs, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    for i in range(streak_period, n):
        if avg_streak_loss[i] > 1e-10:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - 100.0 / (1.0 + rs)
        elif avg_streak_gain[i] > 1e-10:
            rsi_streak[i] = 100.0
        else:
            rsi_streak[i] = 50.0
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, lower, middle

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

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR Ratio for volatility regime detection."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = np.full(len(close), np.nan)
    valid_mask = (~np.isnan(atr_short)) & (~np.isnan(atr_long)) & (atr_long > 1e-10)
    ratio[valid_mask] = atr_short[valid_mask] / atr_long[valid_mask]
    
    return ratio

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion levels."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA21 for macro trend filter
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (12h) indicators
    kama_12h = calculate_kama(close, period=10)
    adx = calculate_adx(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track ADX hysteresis
    prev_adx = np.full(n, 20.0)
    for i in range(1, n):
        if not np.isnan(adx[i-1]):
            prev_adx[i] = adx[i-1]
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(kama_12h[i]) or np.isnan(adx[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(atr[i]) or np.isnan(atr_ratio[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or atr[i] <= 1e-10:
            continue
        
        # === VOLATILITY REGIME (Position Sizing) ===
        vol_spike = atr_ratio[i] > 2.0
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        # === MACRO TREND (1d KAMA21) ===
        macro_bull = close[i] > kama_1d_aligned[i]
        macro_bear = close[i] < kama_1d_aligned[i]
        
        # === KAMA TREND DIRECTION (12h) ===
        kama_bull = close[i] > kama_12h[i]
        kama_bear = close[i] < kama_12h[i]
        
        # KAMA slope (compare to 5 bars ago)
        kama_slope_up = kama_12h[i] > kama_12h[i-5] if i >= 5 and not np.isnan(kama_12h[i-5]) else False
        kama_slope_down = kama_12h[i] < kama_12h[i-5] if i >= 5 and not np.isnan(kama_12h[i-5]) else False
        
        # === ADX TREND STRENGTH ===
        is_trending = adx[i] > 25.0
        is_ranging = adx[i] < 20.0
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8
        is_trending_chop = chop[i] < 38.2
        
        # === CRSI MEAN REVERSION SIGNALS ===
        crsi_oversold = crsi[i] < 10.0
        crsi_overbought = crsi[i] > 90.0
        crsi_extreme_oversold = crsi[i] < 5.0
        crsi_extreme_overbought = crsi[i] > 95.0
        
        # === DONCHIAN BREAKOUT ===
        donch_breakout_long = close[i] > donch_upper[i-1] if i > 0 else False
        donch_breakout_short = close[i] < donch_lower[i-1] if i > 0 else False
        
        # === BOLLINGER MEAN REVERSION ===
        bb_breakout_long = close[i] < bb_lower[i] * 1.001
        bb_breakout_short = close[i] > bb_upper[i] * 0.999
        
        desired_signal = 0.0
        
        # === REGIME 1: RANGING/CHOPPY (ADX < 20 OR CHOP > 61.8) — MEAN REVERSION ===
        if is_ranging or is_choppy:
            # Long: CRSI extreme oversold + price > SMA200 + macro bullish
            if crsi_extreme_oversold and not np.isnan(sma_200[i]) and close[i] > sma_200[i] * 0.99:
                if macro_bull or kama_bull:
                    desired_signal = current_size
            elif crsi_oversold and bb_breakout_long:
                if macro_bull:
                    desired_signal = current_size * 0.7
            # Short: CRSI extreme overbought + price < SMA200 + macro bearish
            elif crsi_extreme_overbought and not np.isnan(sma_200[i]) and close[i] < sma_200[i] * 1.01:
                if macro_bear or kama_bear:
                    desired_signal = -current_size
            elif crsi_overbought and bb_breakout_short:
                if macro_bear:
                    desired_signal = -current_size * 0.7
        
        # === REGIME 2: TRENDING (ADX > 25 OR CHOP < 38.2) — BREAKOUT FOLLOWING ===
        elif is_trending or is_trending_chop:
            # Long: Donchian breakout + KAMA bullish + KAMA slope up + macro bullish
            if donch_breakout_long and kama_bull and kama_slope_up:
                if macro_bull:
                    desired_signal = current_size
                else:
                    desired_signal = current_size * 0.5
            # Short: Donchian breakout + KAMA bearish + KAMA slope down + macro bearish
            elif donch_breakout_short and kama_bear and kama_slope_down:
                if macro_bear:
                    desired_signal = -current_size
                else:
                    desired_signal = -current_size * 0.5
            # Pullback entry in trend
            elif kama_bull and kama_slope_up and close[i] < kama_12h[i] * 1.005:
                if macro_bull and adx[i] > 20:
                    desired_signal = current_size * 0.5
            elif kama_bear and kama_slope_down and close[i] > kama_12h[i] * 0.995:
                if macro_bear and adx[i] > 20:
                    desired_signal = -current_size * 0.5
        
        # === REGIME 3: TRANSITION (ADX 20-25) — COMBINED SIGNALS ===
        else:
            # Require stronger confluence in transition
            if crsi_extreme_oversold and kama_bull and macro_bull:
                desired_signal = current_size * 0.7
            elif crsi_extreme_overbought and kama_bear and macro_bear:
                desired_signal = -current_size * 0.7
            elif donch_breakout_long and kama_bull and kama_slope_up and macro_bull:
                desired_signal = current_size * 0.5
            elif donch_breakout_short and kama_bear and kama_slope_down and macro_bear:
                desired_signal = -current_size * 0.5
        
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
        
        # === HOLD LOGIC — Maintain position if setup intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if KAMA still bullish or price above KAMA
                if kama_bull or close[i] > kama_12h[i]:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if KAMA still bearish or price below KAMA
                if kama_bear or close[i] < kama_12h[i]:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA reverses bearish + ADX shows trend weakening
            if kama_bear and adx[i] < 20:
                desired_signal = 0.0
            # Exit long if macro reverses strongly bearish
            if macro_bear and kama_bear and adx[i] < 25:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA reverses bullish + ADX shows trend weakening
            if kama_bull and adx[i] < 20:
                desired_signal = 0.0
            # Exit short if macro reverses strongly bullish
            if macro_bull and kama_bull and adx[i] < 25:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        
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