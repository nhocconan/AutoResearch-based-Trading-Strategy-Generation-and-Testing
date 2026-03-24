#!/usr/bin/env python3
"""
Experiment #514: 1d Primary + 1w HTF — Regime-Adaptive Dual Strategy

Hypothesis: Daily timeframe with weekly HTF provides optimal signal quality
for crypto perpetuals. Strategy switches between mean-reversion (choppy markets)
and trend-following (trending markets) based on Choppiness Index.

Key innovations:
1. 1w HMA(21) = major trend bias (HTF filter) - call ONCE before loop
2. 1d Choppiness(14) = regime detection (CHOP>55 range, CHOP<45 trend)
3. 1d Connors RSI = mean reversion entry timing (CRSI<20 long, CRSI>80 short)
4. 1d Donchian(20) = breakout confirmation for trend regime
5. 1d ATR(14)*2.5 = trailing stoploss on all positions
6. Regime-adaptive: different logic per market state

Why this should work:
- 1d timeframe = 20-50 trades/year target (fee-efficient)
- Weekly HTF filters out noise and major trend direction
- Choppiness regime switch adapts to market state (bull/bear/range)
- Connors RSI proven on ETH (Sharpe +0.923 in research)
- Conservative sizing (0.25-0.30) survives 2022-style crashes

Target: Sharpe>0.40, trades>=80 train, trades>=10 test, DD>-40%
Timeframe: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_crsi_donchian_1w_v2"
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
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3) of close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI(2) of streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    ema_12 = calculate_ema(close, period=12)
    ema_26 = calculate_ema(close, period=26)
    atr = calculate_atr(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # MACD histogram
    macd_line = ema_12 - ema_26
    macd_signal = pd.Series(macd_line).ewm(span=9, min_periods=9, adjust=False).mean().values
    macd_hist = macd_line - macd_signal
    
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1w HTF BIAS ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d HMA TREND ===
        hma_bull = close[i] > hma_21[i]
        hma_bear = close[i] < hma_21[i]
        hma_cross_bull = hma_21[i] > hma_50[i] if not np.isnan(hma_50[i]) else False
        hma_cross_bear = hma_21[i] < hma_50[i] if not np.isnan(hma_50[i]) else False
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0
        chop_trend = chop[i] < 45.0
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 and not np.isnan(crsi[i-1]) else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 and not np.isnan(crsi[i-1]) else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakdown_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === MACD MOMENTUM ===
        macd_bull = macd_hist[i] > 0.0 and not np.isnan(macd_hist[i])
        macd_bear = macd_hist[i] < 0.0 and not np.isnan(macd_hist[i])
        macd_cross_up = macd_hist[i] > 0.0 and macd_hist[i-1] <= 0.0 if i > 0 and not np.isnan(macd_hist[i]) and not np.isnan(macd_hist[i-1]) else False
        macd_cross_down = macd_hist[i] < 0.0 and macd_hist[i-1] >= 0.0 if i > 0 and not np.isnan(macd_hist[i]) and not np.isnan(macd_hist[i-1]) else False
        
        # === VOLATILITY FILTER ===
        if i >= 100:
            atr_mean = np.nanmean(atr[max(0,i-100):i])
            atr_ratio = atr[i] / atr_mean if atr_mean > 1e-10 else 1.0
        else:
            atr_ratio = 1.0
        vol_normal = atr_ratio < 3.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow HTF direction with Donchian breakout + MACD
        if chop_trend and vol_normal:
            # Long: HTF bull + Donchian breakout + MACD positive + above SMA50
            if htf_bull and donchian_breakout_long and macd_bull and above_sma50:
                desired_signal = SIZE_STRONG
            # Short: HTF bear + Donchian breakdown + MACD negative + below SMA50
            elif htf_bear and donchian_breakdown_short and macd_bear and below_sma50:
                desired_signal = -SIZE_STRONG
            # HMA crossover confirmation
            elif htf_bull and hma_cross_bull and macd_cross_up and above_sma50:
                desired_signal = SIZE_BASE
            elif htf_bear and hma_cross_bear and macd_cross_down and below_sma50:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean reversion with CRSI extremes
        if chop_range and vol_normal:
            # Long: CRSI extreme oversold + above SMA200 (bullish bias)
            if crsi_extreme_oversold and above_sma200:
                desired_signal = SIZE_BASE
            # Short: CRSI extreme overbought + below SMA200 (bearish bias)
            elif crsi_extreme_overbought and below_sma200:
                desired_signal = -SIZE_BASE
            # CRSI recovery from moderate extremes
            elif crsi_oversold and crsi_rising and above_sma50 and htf_bull:
                desired_signal = SIZE_BASE * 0.8
            elif crsi_overbought and crsi_falling and below_sma50 and htf_bear:
                desired_signal = -SIZE_BASE * 0.8
        
        # NEUTRAL REGIME (chop 45-55): Conservative entries
        if not chop_range and not chop_trend and vol_normal:
            if htf_bull and crsi_oversold and crsi_rising and above_sma50:
                desired_signal = SIZE_BASE
            elif htf_bear and crsi_overbought and crsi_falling and below_sma50:
                desired_signal = -SIZE_BASE
            elif donchian_breakout_long and htf_bull and macd_bull:
                desired_signal = SIZE_BASE * 0.8
            elif donchian_breakdown_short and htf_bear and macd_bear:
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
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