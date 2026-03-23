#!/usr/bin/env python3
"""
Experiment #493: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + Donchian Breakout

Hypothesis: Daily timeframe with weekly HTF bias provides optimal trade quality for crypto.
Key innovation: Choppiness Index (CHOP) detects regime, then applies DIFFERENT logic:
- CHOP > 61.8 (choppy/range): Use Connors RSI mean-reversion at extremes
- CHOP < 38.2 (trending): Use Donchian breakout with HMA trend filter

This dual-regime approach adapts to market conditions instead of using one rigid strategy.
1w HTF provides major trend bias to filter counter-trend trades.

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
Trade frequency: 20-50/year on 1d (low fee drag, high quality signals)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_crsi_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = choppy/ranging market (mean-reversion preferred)
    CHOP < 38.2 = trending market (trend-following preferred)
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Rolling sum of ATR and High/Low range
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean-reversion indicator.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    CRSI < 10 = extremely oversold (long opportunity)
    CRSI > 90 = extremely overbought (short opportunity)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(close, 3)
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    gain_s = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI(streak, 2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] >= 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_gain_s = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_s = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = streak_gain_s / (streak_loss_s + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(close, 100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i - rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    with np.errstate(invalid='ignore'):
        crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA) - reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(series, span):
        """Weighted Moving Average"""
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            weights = np.arange(1, span + 1)
            result[i] = np.sum(series[i - span + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    if half < 1 or sqrt_n < 1:
        return hma
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    raw_hma = np.zeros(n)
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            raw_hma[i] = 2.0 * wma_half[i] - wma_full[i]
        else:
            raw_hma[i] = np.nan
    
    # WMA of raw_hma with sqrt(n) period
    hma = wma(raw_hma, sqrt_n)
    
    return hma

def calculate_donchian(high, low, period=20):
    """
    Donchian Channels - breakout detection.
    Upper = Highest High over period
    Lower = Lowest Low over period
    """
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    crsi_1d = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_1d = calculate_hma(close, period=21)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators (1w HMA for major trend bias)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need 150 bars for CRSI rank_period + donchian
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(chop_1d[i]):
            continue
        if np.isnan(crsi_1d[i]):
            continue
        if np.isnan(hma_1d[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === HTF MAJOR TREND BIAS (1w HMA) ===
        htf_bullish = close[i] > hma_1w_aligned[i]
        htf_bearish = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_1d[i] > 61.8  # Range/mean-reversion regime
        is_trending = chop_1d[i] < 38.2  # Trend-following regime
        # 38.2 - 61.8 is neutral/transition
        
        # === PRIMARY TREND (1d HMA) ===
        price_above_hma = close[i] > hma_1d[i]
        price_below_hma = close[i] < hma_1d[i]
        hma_slope_up = hma_1d[i] > hma_1d[i - 5] if i >= 5 else False
        hma_slope_down = hma_1d[i] < hma_1d[i - 5] if i >= 5 else False
        
        # === CONNORS RSI EXTREMES (for mean-reversion) ===
        crsi_oversold = crsi_1d[i] < 15.0  # Extremely oversold
        crsi_overbought = crsi_1d[i] > 85.0  # Extremely overbought
        
        # === DONCHIAN BREAKOUT (for trend-following) ===
        donchian_breakout_long = close[i] > donchian_upper[i - 1] if i >= 1 else False
        donchian_breakout_short = close[i] < donchian_lower[i - 1] if i >= 1 else False
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        long_conditions = 0
        
        if is_choppy:
            # MEAN-REVERSION MODE: Buy oversold in choppy market
            if crsi_oversold:
                long_conditions += 2  # Strong signal
            if htf_bullish:
                long_conditions += 1  # HTF bias support
            if price_above_hma:
                long_conditions += 1  # Short-term trend support
            
            # Enter long in choppy market if CRSI oversold + at least one confirm
            if crsi_oversold and long_conditions >= 3:
                desired_signal = SIZE_LONG
        
        elif is_trending:
            # TREND-FOLLOWING MODE: Buy breakout in trending market
            if donchian_breakout_long:
                long_conditions += 2  # Strong breakout signal
            if htf_bullish:
                long_conditions += 2  # HTF bias strongly supports
            if hma_slope_up:
                long_conditions += 1  # Momentum confirmation
            
            # Enter long in trending market if breakout + HTF bias
            if donchian_breakout_long and long_conditions >= 4:
                desired_signal = SIZE_LONG
        
        else:
            # NEUTRAL/TRANSITION: Conservative entries only
            if crsi_oversold and htf_bullish and price_above_hma:
                desired_signal = SIZE_LONG
        
        # === SHORT ENTRIES ===
        if desired_signal == 0.0:
            short_conditions = 0
            
            if is_choppy:
                # MEAN-REVERSION MODE: Sell overbought in choppy market
                if crsi_overbought:
                    short_conditions += 2
                if htf_bearish:
                    short_conditions += 1
                if price_below_hma:
                    short_conditions += 1
                
                if crsi_overbought and short_conditions >= 3:
                    desired_signal = -SIZE_SHORT
            
            elif is_trending:
                # TREND-FOLLOWING MODE: Sell breakout in trending market
                if donchian_breakout_short:
                    short_conditions += 2
                if htf_bearish:
                    short_conditions += 2
                if hma_slope_down:
                    short_conditions += 1
                
                if donchian_breakout_short and short_conditions >= 4:
                    desired_signal = -SIZE_SHORT
            
            else:
                # NEUTRAL/TRANSITION: Conservative entries only
                if crsi_overbought and htf_bearish and price_below_hma:
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
                # Hold long if HTF still bullish OR price above HMA
                if htf_bullish or price_above_hma:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF still bearish OR price below HMA
                if htf_bearish or price_below_hma:
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
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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