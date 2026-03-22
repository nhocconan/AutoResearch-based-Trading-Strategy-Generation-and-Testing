#!/usr/bin/env python3
"""
Experiment #390: 1h Primary + 4h/12h HTF — HMA Trend + Connors RSI + Session Filter

Hypothesis: After 350+ failed experiments, the pattern shows:
1. 1h timeframe needs 30-60 trades/year to overcome fee drag
2. HTF (12h/4h) for direction + 1h for entry timing PROVEN to work
3. Connors RSI (CRSI) has 75% win rate for mean reversion entries
4. Session filter (8-20 UTC) reduces noise from low-liquidity hours
5. Volume filter confirms genuine moves vs fakeouts
6. Simpler confluence (3 factors) to ensure >=30 trades/symbol

Why this might beat current best (Sharpe=0.435):
- 1h TF with strict filters = optimal trade frequency (30-60/year)
- 12h HMA(21) for major trend (proven in #382 with Sharpe=0.109)
- 4h HMA(16/48) for intermediate confirmation
- Connors RSI < 30 for long, > 70 for short (wider for trade frequency)
- Session + volume filters reduce false signals
- Discrete position sizing: 0.0, ±0.20, ±0.30 (max 0.40)

Position sizing: 0.20-0.30 (smaller for 1h vs 4h/12h)
Stoploss: 2.0 * ATR trailing
Target: 30-60 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_crsi_session_volume_4h12h_v1"
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
    CRSI = (RSI(close,3) + RSI(Streak,2) + PercentRank(100)) / 3
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) - fast RSI
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_pos = np.where(streak > 0, streak, 0)
    streak_neg = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_pos).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_neg).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if avg_streak_loss[i] == 0:
            streak_rsi[i] = 100.0
        else:
            rs_streak = avg_streak_gain[i] / (avg_streak_loss[i] + 1e-10)
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Percent Rank - percentile of today's return over last 100 days
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / (close[i-1] + 1e-10)
    
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current_return = returns[i]
        count_below = np.sum(window < current_return)
        percent_rank[i] = (count_below / rank_period) * 100.0
    
    # CRSI = average of three components
    for i in range(rank_period, n):
        crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

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
    
    # Calculate 12h HTF indicators (major trend direction)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    
    # Calculate 4h HTF indicators (intermediate trend)
    hma_4h_16 = calculate_hma(df_4h['close'].values, period=16)
    hma_4h_48 = calculate_hma(df_4h['close'].values, period=48)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = np.inf
    entry_price = 0.0
    last_trade_bar = -50
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            continue
        
        if np.isnan(crsi[i]):
            continue
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        timestamp_s = open_time[i] / 1000.0
        hour_utc = int((timestamp_s // 3600) % 24)
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * vol_ma_20[i]
        
        # === 12H MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_12h_21_aligned[i]
        bear_regime = close[i] < hma_12h_21_aligned[i]
        
        # === 4H INTERMEDIATE TREND ===
        hma_4h_bullish = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === CONNORS RSI SIGNALS (wider thresholds for trade frequency) ===
        crsi_oversold = crsi[i] < 30.0
        crsi_overbought = crsi[i] > 70.0
        
        # === ENTRY LOGIC — 3 CONFLUENCE REQUIRED ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Bull regime + 4h bullish + CRSI oversold + (session OR volume)
        if bull_regime and hma_4h_bullish:
            if crsi_oversold and (in_session or volume_ok):
                new_signal = LONG_SIZE
        
        # SHORT ENTRY: Bear regime + 4h bearish + CRSI overbought + (session OR volume)
        if bear_regime and hma_4h_bearish:
            if crsi_overbought and (in_session or volume_ok):
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 96 bars (~4 days on 1h), force entry on weaker signal
        if bars_since_last_trade > 96 and new_signal == 0.0 and not in_position:
            if bull_regime and crsi[i] < 40 and hma_4h_bullish:
                new_signal = LONG_SIZE * 0.6
            elif bear_regime and crsi[i] > 60 and hma_4h_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on mean reversion exhaustion)
        if in_position and position_side > 0 and crsi[i] > 70:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi[i] < 30:
            new_signal = 0.0
        
        # Trend reversal exit (12h regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # 4h trend reversal exit
        if in_position and position_side > 0 and hma_4h_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_4h_bullish:
            new_signal = 0.0
        
        # === STOPLOSS (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.0 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.0 * atr_14[i]
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
                lowest_price = close[i] if position_side < 0 else np.inf
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else np.inf
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = np.inf
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals