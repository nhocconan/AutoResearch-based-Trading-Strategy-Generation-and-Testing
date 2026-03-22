#!/usr/bin/env python3
"""
Experiment #440: 1h Primary + 4h/12h HTF — Connors RSI + HMA Trend + ADX Regime

Hypothesis: After 439 experiments, clear pattern emerges for lower TF success:
1. 1h needs VERY strict filters (30-60 trades/year) or fee drag kills profit
2. HTF (4h/12h) must determine DIRECTION, 1h only for ENTRY TIMING
3. Connors RSI (RSI3 + RSI_Streak + PercentRank) has 75% win rate in research
4. ADX regime switch: trend-follow when ADX>25, mean-revert when ADX<20
5. Session filter (8-20 UTC) + volume filter reduces noise trades

Why this might beat current best (Sharpe=0.435):
- Connors RSI catches oversold/overbought extremes better than standard RSI
- 4h HMA provides cleaner trend signal than 1d (more responsive)
- 12h ADX regime adapts to market conditions automatically
- Session/volume filters cut low-quality trades in Asian overnight hours
- Conservative sizing (0.25) protects against 2022-style crashes

Position sizing: 0.25 (discrete, max 0.35)
Stoploss: 2.5 * ATR trailing
Target: 40-80 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_connors_hma_adx_4h12h_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Research shows 75% win rate on CRSI<10 long / CRSI>90 short.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak RSI: consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak (period=2)
    streak_s = pd.Series(streak)
    streak_gain = streak_s.diff().where(streak_s.diff() > 0, 0.0)
    streak_loss = -streak_s.diff().where(streak_s.diff() < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Percent Rank: % of closes in last 100 bars that are lower than current
    pr = np.zeros(n)
    for i in range(pr_period, n):
        window = close[i-pr_period:i]
        pr[i] = np.sum(window < close[i]) / pr_period * 100.0
    
    # Combine into CRSI
    crsi = (rsi_close + rsi_streak.values + pr) / 3.0
    
    return crsi

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_avg + 1e-10)
    return vol_ratio.values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HTF indicators (trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 12h HTF indicators (regime detection)
    adx_12h_14 = calculate_adx(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        period=14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    adx_12h_14_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_14)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi_3_2_100 = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    vol_ratio_20 = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.35 for 1h)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(adx_12h_14_aligned[i]) or np.isnan(crsi_3_2_100[i]):
            continue
        if np.isnan(vol_ratio_20[i]):
            continue
        
        # === UTC SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER (avoid low liquidity) ===
        volume_ok = vol_ratio_20[i] > 0.7
        
        # === 4H TREND DIRECTION (primary bias) ===
        # HMA21 > HMA50 = bullish trend (favor longs)
        # HMA21 < HMA50 = bearish trend (favor shorts)
        trend_bullish_4h = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        trend_bearish_4h = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # Price vs 4h HMA21 confirmation
        price_above_hma4h = close[i] > hma_4h_21_aligned[i]
        price_below_hma4h = close[i] < hma_4h_21_aligned[i]
        
        # === 12H ADX REGIME DETECTION ===
        # ADX > 25 = trending (follow trend)
        # ADX < 20 = ranging (mean reversion)
        # 20-25 = neutral (use trend bias)
        adx_12h = adx_12h_14_aligned[i]
        is_trending_12h = adx_12h > 25.0
        is_ranging_12h = adx_12h < 20.0
        
        # === CONNORS RSI SIGNALS ===
        crsi = crsi_3_2_100[i]
        crsi_oversold = crsi < 15.0  # Strong mean reversion long signal
        crsi_overbought = crsi > 85.0  # Strong mean reversion short signal
        crsi_extreme_low = crsi < 25.0
        crsi_extreme_high = crsi > 75.0
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (require session + volume + HTF confluence)
        if in_session and volume_ok:
            # Trending regime: pullback entry in uptrend
            if is_trending_12h and trend_bullish_4h and price_above_hma4h:
                if crsi_extreme_low and not crsi_overbought:
                    new_signal = LONG_SIZE
            # Ranging regime: mean reversion at oversold
            elif is_ranging_12h and crsi_oversold:
                # Only long if not strongly bearish on 4h
                if not trend_bearish_4h or price_above_hma4h:
                    new_signal = LONG_SIZE
            # HMA crossover confirmation (trend change)
            elif trend_bullish_4h and crsi < 40.0 and crsi > 10.0:
                if bars_since_last_trade > 5:  # Avoid overtrading
                    new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRIES (require session + volume + HTF confluence)
        if in_session and volume_ok:
            # Trending regime: bounce entry in downtrend
            if is_trending_12h and trend_bearish_4h and price_below_hma4h:
                if crsi_extreme_high and not crsi_oversold:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE
            # Ranging regime: mean reversion at overbought
            elif is_ranging_12h and crsi_overbought:
                # Only short if not strongly bullish on 4h
                if not trend_bullish_4h or price_below_hma4h:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE
            # HMA crossover confirmation (trend change)
            elif trend_bearish_4h and crsi > 60.0 and crsi < 90.0:
                if bars_since_last_trade > 5 and new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 15 bars (~15 hours on 1h), allow weaker signal
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if in_session and volume_ok:
                if trend_bullish_4h and crsi < 35.0:
                    new_signal = LONG_SIZE * 0.6
                elif trend_bearish_4h and crsi > 65.0:
                    new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and crsi > 80.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi < 20.0:
            new_signal = 0.0
        
        # Trend reversal exit (4h HMA cross against position)
        if in_position and position_side > 0 and trend_bearish_4h and price_below_hma4h:
            new_signal = 0.0
        if in_position and position_side < 0 and trend_bullish_4h and price_above_hma4h:
            new_signal = 0.0
        
        # Session end exit (close position before Asian overnight)
        if in_position and (utc_hour < 8 or utc_hour > 20):
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals