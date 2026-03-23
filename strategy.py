#!/usr/bin/env python3
"""
Experiment #634: 4h Primary + 12h/1d HTF — Connors RSI + Donchian + Choppiness Regime

Hypothesis: 4h timeframe with 12h/1d HTF filter provides optimal balance between
signal frequency (20-50 trades/year) and signal quality. Connors RSI excels at
mean-reversion entries in choppy markets, while Donchian breakouts capture trends.
Choppiness Index switches between regimes dynamically.

Key innovations:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Proven 75% win rate on mean-reversion trades
   - Long when CRSI < 15, Short when CRSI > 85
2. Donchian Channel(20) breakout with volume confirmation
   - Break above 20-day high + volume > 1.5x avg = trend long
   - Break below 20-day low + volume > 1.5x avg = trend short
3. Choppiness Index regime switch
   - CHOP > 55 = choppy (use CRSI mean-reversion)
   - CHOP < 45 = trending (use Donchian breakout)
4. 12h HMA(21) for macro trend bias
   - Only long when price > 12h HMA
   - Only short when price < 12h HMA
5. ADX(14) > 20 for trend strength confirmation
6. ATR(14) trailing stop at 2.5x for risk management

Why this should beat Sharpe=0.612:
- CRSI has documented edge in bear/range markets (2022, 2025)
- Dual-regime approach adapts to market conditions
- 4h TF = fewer false signals than 1h, more trades than 12h
- Volume confirmation filters false breakouts
- Conservative sizing (0.25-0.30) survives crashes

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_donchian_chop_regime_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI).
    Combines 3 components for mean-reversion signals:
    1. RSI(3) on close
    2. RSI(2) on streak (consecutive up/down days)
    3. PercentRank(100) of close vs past 100 bars
    
    CRSI = (RSI_close + RSI_streak + PercentRank) / 3
    Long signal: CRSI < 15
    Short signal: CRSI > 85
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # Component 1: RSI(3) on close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100 - (100 / (1 + rs))
        rsi_close = np.clip(rsi_close, 0, 100)
    
    # Component 2: RSI(2) on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to binary up/down for RSI
    streak_up = np.where(streak > 0, np.abs(streak), 0)
    streak_down = np.where(streak < 0, np.abs(streak), 0)
    
    avg_streak_gain = pd.Series(streak_up).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_down).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + rs_streak))
        rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window[:-1] < close[i]) / (rank_period - 1)
        percent_rank[i] = rank * 100
    
    # Combine components
    valid_mask = (~np.isnan(rsi_close)) & (~np.isnan(rsi_streak)) & (~np.isnan(percent_rank))
    crsi[valid_mask] = (rsi_close[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel.
    Upper = highest high over period
    Lower = lowest low over period
    Middle = (Upper + Lower) / 2
    """
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use: > 55 = chop (mean revert), < 45 = trend (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Sum ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
        chop = np.clip(chop_raw, 0, 100)
    
    return chop

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX).
    Measures trend strength (not direction).
    ADX > 25 = strong trend, ADX < 20 = weak/no trend
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        diff_high = high[i] - high[i-1]
        diff_low = low[i-1] - low[i]
        
        if diff_high > diff_low and diff_high > 0:
            plus_dm[i] = diff_high
        if diff_low > diff_high and diff_low > 0:
            minus_dm[i] = diff_low
    
    # Smooth with Wilder's method (EMA with span=period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    # DX and ADX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother HTF trend."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average for volume confirmation."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h indicators (primary timeframe)
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    adx_4h = calculate_adx(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    vol_ma_4h = calculate_volume_ma(volume, period=20)
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(chop_4h[i]) or np.isnan(adx_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(vol_ma_4h[i]) or vol_ma_4h[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_4h[i] > 55.0
        is_trending = chop_4h[i] < 45.0
        
        # === HTF TREND BIAS (12h HMA) ===
        htf_12h_bullish = close[i] > hma_12h_aligned[i]
        htf_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx_4h[i] > 20.0
        
        # === VOLUME CONFIRMATION ===
        high_volume = volume[i] > 1.5 * vol_ma_4h[i]
        
        # === CRSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi_4h[i] < 15.0
        crsi_overbought = crsi_4h[i] > 85.0
        
        # === DONCHIAN BREAKOUT SIGNALS (Trend) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion with CRSI) ===
        if is_choppy:
            # Long: CRSI oversold + HTF 12h not strongly bearish
            if crsi_oversold and not htf_12h_bearish:
                desired_signal = SIZE_LONG
            # Short: CRSI overbought + HTF 12h not strongly bullish
            elif crsi_overbought and not htf_12h_bullish:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: TRENDING MARKET (Trend Follow with Donchian) ===
        elif is_trending:
            # Long: Donchian breakout + high volume + HTF bullish + ADX strong
            if donchian_breakout_long and high_volume and htf_12h_bullish and strong_trend:
                desired_signal = SIZE_LONG
            # Short: Donchian breakout + high volume + HTF bearish + ADX strong
            elif donchian_breakout_short and high_volume and htf_12h_bearish and strong_trend:
                desired_signal = -SIZE_SHORT
            # Fallback: HTF trend direction with ADX confirmation
            elif htf_12h_bullish and strong_trend and crsi_4h[i] < 50:
                desired_signal = SIZE_LONG
            elif htf_12h_bearish and strong_trend and crsi_4h[i] > 50:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 3: NEUTRAL/TRANSITION ===
        else:
            # Use HTF direction with CRSI filter
            if htf_12h_bullish and crsi_oversold:
                desired_signal = SIZE_LONG
            elif htf_12h_bearish and crsi_overbought:
                desired_signal = -SIZE_SHORT
            elif htf_12h_bullish and crsi_4h[i] < 40:
                desired_signal = SIZE_LONG
            elif htf_12h_bearish and crsi_4h[i] > 60:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF still bullish OR CRSI not extremely overbought
                if htf_12h_bullish and crsi_4h[i] < 80:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF still bearish OR CRSI not extremely oversold
                if htf_12h_bearish and crsi_4h[i] > 20:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
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
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
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