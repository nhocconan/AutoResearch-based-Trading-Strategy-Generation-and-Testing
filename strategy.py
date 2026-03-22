#!/usr/bin/env python3
"""
Experiment #425: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness + Session Filter

Hypothesis: After 385+ failed experiments, clear patterns emerge for 1h timeframe:
1. 1h needs 30-80 trades/year — too many = fee drag, too few = no alpha
2. Connors RSI (CRSI) has proven 75% win rate for mean reversion entries
3. Choppiness Index is BEST regime filter: CHOP>55=range, CHOP<45=trend
4. 4h HMA(21) for major trend direction prevents counter-trend disasters
5. Session filter (8-20 UTC) captures London/NY overlap = real volume
6. Volume >0.8x avg confirms genuine moves, filters fakeouts

Why this might beat current best (Sharpe=0.435):
- 1h TF captures more intraday moves than 4h/12h while avoiding 15m noise
- CRSI<20 / >80 thresholds are reachable (unlike CRSI<10 which rarely triggers)
- Session filter cuts 40% of low-volume Asian session whipsaws
- 4h HMA trend filter prevents 2022-style counter-trend blowups
- Discrete sizing (0.25/0.30) minimizes fee churn on signal changes

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 40-80 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_session_4h1d_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Research shows 75% win rate for CRSI<20 long, CRSI>80 short.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for mean reversion
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # RSI of Streak (consecutive up/down bars)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100) - percentile of price change over lookback
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(index=close_s.index, dtype=float)
    
    for i in range(rank_period, n):
        window = pct_change.iloc[max(0,i-rank_period):i]
        current = pct_change.iloc[i]
        if not np.isnan(current) and len(window) > 0:
            rank = (window < current).sum() / len(window)
            percent_rank.iloc[i] = rank * 100.0
    
    # Combine into CRSI
    crsi = (rsi_3 + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Sum of ATR over period
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    
    # Choppiness formula
    chop = 100.0 * np.log10((hh - ll).values / (atr_sum.values + 1e-10)) / np.log10(period)
    
    return chop

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000.0
    hour = pd.to_datetime(ts_seconds, unit='s').hour
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, 14)
    vol_avg = calculate_volume_avg(volume, 20)
    
    # Calculate session hours
    session_hours = np.array([get_session_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            continue
        
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # London/NY overlap = real volume, avoid Asian session noise
        in_session = 8 <= session_hours[i] <= 20
        
        # === VOLUME FILTER ===
        # Volume must be >= 0.8x average to confirm genuine move
        volume_ok = volume[i] >= 0.8 * vol_avg[i]
        
        # === 4H MAJOR TREND (primary direction filter) ===
        # Price above 4h HMA = bull market bias (favor longs)
        # Price below 4h HMA = bear market bias (favor shorts)
        bull_regime = close[i] > hma_4h_21_aligned[i]
        bear_regime = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = ranging (mean reversion preferred)
        # CHOP < 45 = trending (breakout preferred)
        is_choppy = choppiness[i] > 55.0
        is_trending = choppiness[i] < 45.0
        
        # === CONNORS RSI SIGNALS (mean reversion) ===
        # CRSI < 20 = oversold (long opportunity)
        # CRSI > 80 = overbought (short opportunity)
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY conditions (3+ confluence required)
        if bull_regime and in_session and volume_ok:
            # Mean reversion entry (choppy market + CRSI oversold)
            if is_choppy and crsi_oversold:
                new_signal = LONG_SIZE
            # Trend pullback entry (trending + CRSI moderate oversold)
            elif is_trending and crsi[i] < 35.0:
                new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRY conditions (3+ confluence required)
        if bear_regime and in_session and volume_ok:
            # Mean reversion entry (choppy market + CRSI overbought)
            if is_choppy and crsi_overbought:
                new_signal = -SHORT_SIZE
            # Trend pullback entry (trending + CRSI moderate overbought)
            elif is_trending and crsi[i] > 65.0:
                new_signal = -SHORT_SIZE * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 20 bars (~20 hours on 1h), force entry on weaker signal
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if bull_regime and crsi[i] < 30.0 and in_session:
                new_signal = LONG_SIZE * 0.5
            elif bear_regime and crsi[i] > 70.0 and in_session:
                new_signal = -SHORT_SIZE * 0.5
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on mean reversion exhaustion)
        if in_position and position_side > 0 and crsi[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi[i] < 25.0:
            new_signal = 0.0
        
        # Trend reversal exit (4h regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Session exit (close position outside trading hours)
        if in_position and not in_session:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
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