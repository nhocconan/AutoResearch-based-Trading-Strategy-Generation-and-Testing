#!/usr/bin/env python3
"""
Experiment #260: 1h Primary + 4h/12h HTF — Connors RSI + HMA Trend + Choppiness Regime

Hypothesis: After #250 failed (Sharpe=-1.946) on 1h, the issue was TOO MANY TRADES.
This version uses EXTREMELY STRICT confluence filters to limit trades to 30-60/year:

1. 12h HMA(21) for PRIMARY trend direction (slow, reliable regime filter)
2. 4h HMA(21/50) for secondary confirmation (must align with 12h)
3. 1h Connors RSI for entry timing (RSI3 + StreakRSI2 + PercentRank100) / 3
4. Choppiness(14) regime filter: CHOP>55 = mean revert, CHOP<45 = trend follow
5. Session filter: only trade 8-20 UTC (high volume periods)
6. Volume filter: volume > 0.8x 20-bar average

Key difference from #250: MUCH stricter entry conditions. Need 4+ confluence:
- HTF trend aligned (12h + 4h HMA)
- Regime appropriate (CHOP confirms strategy type)
- CRSI at extreme (<15 or >85)
- Session + volume confirmation

Position sizing: 0.20 base, 0.30 strong (smaller than 4h strategies due to fee drag)
Target: 30-60 trades/year (CRITICAL — lower TF must have fewer trades)
Stoploss: 2.5 * ATR trailing

Why this might work:
- 12h trend filter prevents whipsaw in ranging markets
- Connors RSI has 75% win rate on mean-reversion (literature-backed)
- Session filter avoids low-volume fake breakouts
- Very strict confluence = fewer trades = less fee drag
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_connors_hma_chop_session_4h12h_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak: consecutive days up (+1) or down (-1)
    PercentRank: percentile of today's return vs last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    returns = close_s.pct_change().fillna(0).values
    streak = np.zeros(n)
    for i in range(1, n):
        if returns[i] > 0:
            streak[i] = max(0, streak[i-1] + 1)
        elif returns[i] < 0:
            streak[i] = min(0, streak[i-1] - 1)
        else:
            streak[i] = streak[i-1]
    
    # RSI(2) on streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window_returns = returns[i-rank_period+1:i+1]
        current_return = returns[i]
        rank = np.sum(window_returns < current_return) / len(window_returns) * 100
        percent_rank[i] = rank
    
    # CRSI = average of three components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_session_filter(open_time):
    """
    Return 1 if hour is between 8-20 UTC (high volume session), 0 otherwise.
    open_time is in milliseconds since epoch.
    """
    # Convert to hours UTC
    hours = (open_time // 3600000) % 24
    session_active = (hours >= 8) & (hours <= 20)
    return session_active.astype(float)

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
    
    # Calculate 12h HTF indicators (primary trend regime)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, 50)
    
    # Calculate 4h HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Volume average (20 bars)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter
    session_active = calculate_session_filter(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    consecutive_no_signal = 0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === 12H TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_12h_21_aligned[i] and hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        regime_bear = close[i] < hma_12h_21_aligned[i] and hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        regime_neutral = not regime_bull and not regime_bear
        
        # === 4H CONFIRMATION (must align with 12h) ===
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === SESSION + VOLUME FILTER ===
        in_session = session_active[i] >= 0.5
        vol_confirmed = volume[i] > 0.8 * vol_avg[i]
        
        # === ENTRY LOGIC (VERY STRICT - need 4+ confluence) ===
        new_signal = 0.0
        
        # LONG entries (need: bull regime + 4h confirm + CRSI oversold + session + volume)
        if regime_bull and hma_4h_bullish:
            # Trend-following pullback entry
            if crsi_oversold and in_session and vol_confirmed:
                new_signal = BASE_SIZE
            # Extreme oversold in any regime (mean reversion)
            if crsi_extreme_oversold and in_session:
                new_signal = STRONG_SIZE
        
        # SHORT entries (need: bear regime + 4h confirm + CRSI overbought + session + volume)
        if regime_bear and hma_4h_bearish:
            # Trend-following pullback entry
            if crsi_overbought and in_session and vol_confirmed:
                new_signal = -BASE_SIZE
            # Extreme overbought in any regime (mean reversion)
            if crsi_extreme_overbought and in_session:
                new_signal = -STRONG_SIZE
        
        # === MEAN REVERSION MODE (when choppy) ===
        if is_choppy and not regime_neutral:
            # In choppy market, trade CRSI extremes regardless of trend
            if crsi_extreme_oversold and in_session:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE
            if crsi_extreme_overbought and in_session:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === TRADE FREQUENCY CONTROL (CRITICAL for 1h) ===
        # Only allow new trades if enough bars passed since last trade
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade < 12:  # Minimum 12 hours between trades
            if in_position:
                new_signal = signals[i-1] if i > 0 else 0.0
            else:
                new_signal = 0.0
        
        # Force exit after 48 hours if no progress (time-based exit)
        if in_position and bars_since_last_trade > 48:
            new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
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
                # Position reversal
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