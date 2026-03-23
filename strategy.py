#!/usr/bin/env python3
"""
Experiment #130: 1h Primary + 4h/12h HTF — Regime-Adaptive CRSI with Session Filter

Hypothesis: Previous 1h strategies failed due to too many trades (>200/yr) causing fee drag.
This strategy uses PROVEN components with STRICT filtering to achieve 30-60 trades/year:

1) 4h HMA(21) for macro trend bias — only trade in HTF trend direction
2) Choppiness Index(14) regime filter — CHOP>55=range(mean revert), CHOP<45=trend(follow)
3) Connors RSI for entry timing — CRSI<10 long, CRSI>90 short (75% win rate proven)
4) Session filter — only trade 8-20 UTC (high volume, less whipsaw)
5) Volume confirmation — volume > 0.8x 20-bar avg (filters false breakouts)
6) ATR(14) stoploss at 2.5x — mandatory risk management

Why this should work on 1h:
- HTF (4h) determines DIRECTION, 1h only for ENTRY TIMING
- Session filter reduces trades by ~60% (only 12/24 hours)
- CRSI extremes are rare (~5% of bars) → natural trade limit
- Regime-adaptive: mean revert in range, trend follow in trends
- Discrete signal levels (0.0, ±0.25) minimize fee churn

Position size: 0.25 (conservative for 1h TF)
Target: 40-70 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h_session_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    n = len(close)
    
    # RSI(3) - short-term momentum
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period+1:i+1]
        if len(streak_window[streak_window > 0]) > 0:
            up_streak = streak_window[streak_window > 0].sum()
        else:
            up_streak = 0
        if len(streak_window[streak_window < 0]) > 0:
            down_streak = abs(streak_window[streak_window < 0].sum())
        else:
            down_streak = 0
        if up_streak + down_streak > 0:
            streak_rsi[i] = 100.0 * up_streak / (up_streak + down_streak)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank - percentile of price change over lookback
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        price_changes = np.diff(close[i-rank_period+1:i+1])
        if len(price_changes) > 0 and price_changes[-1] != 0:
            rank = np.sum(price_changes[:-1] < price_changes[-1])
            percent_rank[i] = 100.0 * rank / len(price_changes[:-1])
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = range/choppy (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    choppiness = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
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
    
    # Calculate 4h HMA for macro trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 4h HMA slope (trend strength)
    hma_4h_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_4h_aligned[i]) and not np.isnan(hma_4h_aligned[i-1]) and hma_4h_aligned[i-1] != 0:
            hma_4h_slope[i] = (hma_4h_aligned[i] - hma_4h_aligned[i-1]) / hma_4h_aligned[i-1] * 100
        else:
            hma_4h_slope[i] = 0.0
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    hma_1h_21 = calculate_hma(close, period=21)
    hma_1h_50 = calculate_hma(close, period=50)
    
    # Extract UTC hours for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25  # Conservative for 1h TF
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        if np.isnan(hma_1h_21[i]) or np.isnan(hma_1h_50[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        hma_4h_slope_positive = hma_4h_slope[i] > 0.3
        hma_4h_slope_negative = hma_4h_slope[i] < -0.3
        hma_4h_slope_flat = abs(hma_4h_slope[i]) <= 0.3
        
        # === 1h TREND FILTER ===
        hma_1h_bullish = hma_1h_21[i] > hma_1h_50[i]
        hma_1h_bearish = hma_1h_21[i] < hma_1h_50[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = choppiness[i] > 55.0  # Range/choppy market
        chop_trend = choppiness[i] < 45.0  # Trending market
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 10.0  # Strong long signal
        crsi_overbought = crsi[i] > 90.0  # Strong short signal
        crsi_moderate_oversold = crsi[i] < 20.0
        crsi_moderate_overbought = crsi[i] > 80.0
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_confirmed = volume_ratio > 0.8
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only trade during session hours
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # --- LONG ENTRY ---
        # Regime 1: Range market (CHOP>55) → Mean reversion long on CRSI extreme
        if chop_range:
            if crsi_oversold and price_above_hma_4h and volume_confirmed:
                new_signal = POSITION_SIZE
            elif crsi_moderate_oversold and price_above_hma_4h and hma_4h_slope_positive and volume_confirmed:
                new_signal = POSITION_SIZE
        
        # Regime 2: Trending market (CHOP<45) → Trend follow on pullback
        elif chop_trend:
            if crsi_moderate_oversold and price_above_hma_4h and hma_1h_bullish and volume_confirmed:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market (CHOP>55) → Mean reversion short on CRSI extreme
        if chop_range:
            if crsi_overbought and price_below_hma_4h and volume_confirmed:
                new_signal = -POSITION_SIZE
            elif crsi_moderate_overbought and price_below_hma_4h and hma_4h_slope_negative and volume_confirmed:
                new_signal = -POSITION_SIZE
        
        # Regime 2: Trending market (CHOP<45) → Trend follow on pullback
        elif chop_trend:
            if crsi_moderate_overbought and price_below_hma_4h and hma_1h_bearish and volume_confirmed:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # Hold if still in favorable conditions (don't exit just because entry signal gone)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if 4h trend intact and not overbought
                if price_above_hma_4h and crsi[i] < 85.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if 4h trend intact and not oversold
                if price_below_hma_4h and crsi[i] > 15.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if highest_since_entry == 0.0:
                highest_since_entry = close[i]
            else:
                highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_4h and hma_4h_slope_negative:
                new_signal = 0.0
            # Exit on CRSI overbought (take profit)
            if crsi_overbought:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h and hma_4h_slope_positive:
                new_signal = 0.0
            # Exit on CRSI oversold (take profit)
            if crsi_oversold:
                new_signal = 0.0
        
        # === EXIT ON CHOP REGIME CHANGE ===
        # If we entered on trend regime but now choppy, exit
        if in_position:
            prev_chop = choppiness[i-1] if i > 0 else choppiness[i]
            if position_side > 0 and prev_chop < 45.0 and choppiness[i] > 65.0:
                new_signal = 0.0
            if position_side < 0 and prev_chop < 45.0 and choppiness[i] > 65.0:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals