#!/usr/bin/env python3
"""
Experiment #545: 1h Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + Volume Session

Hypothesis: After 480+ failed strategies, the key insight for 1h timeframe is:
- Lower TF (1h) MUST use HTF (4h/1d) for direction, 1h only for entry timing
- Choppiness Index regime filter works well on 1d (range vs trend detection)
- Connors RSI (CRSI) has 75% win rate for mean reversion entries
- Session filter (8-20 UTC) reduces low-liquidity whipsaws
- Volume confirmation prevents false breakouts
- Target: 40-80 trades/year on 1h (NOT >100 or fee drag kills profit)

This strategy combines:
1. 4h HMA(21) for primary trend direction (HTF filter)
2. 1d Choppiness Index(14) for regime detection (>55=range, <45=trend)
3. Connors RSI(3,2,100) for entry timing (extreme <10 long, >90 short)
4. Volume > 0.8x 20-bar average for confirmation
5. Session filter: only 8-20 UTC (high liquidity hours)
6. ATR(14) 2.5x trailing stop for risk management
7. Asymmetric sizing: 0.25 long, 0.20 short (crypto bias)

Why this might work on 1h:
- 4h trend filter prevents counter-trend trades (major failure mode)
- 1d Choppiness avoids trend-following in ranges and vice versa
- Connors RSI catches pullbacks within HTF trend
- Session filter reduces Asian session whipsaws (low liquidity)
- Volume filter confirms genuine moves vs fakeouts
- 1h TF targets 50-80 trades/year (optimal for this timeframe)

Position sizing: 0.25 long / 0.20 short (discrete, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_connors_rsi_volume_4h1d_session_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) - proven mean reversion indicator.
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) of close - short-term momentum
    2. RSI(2) of streak - consecutive up/down days
    3. PercentRank(100) - where current return ranks vs last 100 bars
    
    Entry signals: CRSI < 10 (oversold long), CRSI > 90 (overbought short)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) of close
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI(2) of streak
    # Streak = consecutive up/down bars (+1 for up, -1 for down, 0 for flat)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Component 3: PercentRank(100) of returns
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        current = returns[i]
        if np.isnan(current):
            percent_rank[i] = 50.0
        else:
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                percent_rank[i] = 100.0 * np.sum(valid_window < current) / len(valid_window)
            else:
                percent_rank[i] = 50.0
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) - regime detection.
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = Range/Consolidation (mean reversion favorable)
    - CHOP < 38.2 = Trending (trend following favorable)
    - 38.2 < CHOP < 61.8 = Transition
    """
    n = len(close)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    hhll = hh - ll
    
    # Choppiness calculation
    chop = np.zeros(n)
    mask = (hhll > 0) & (atr_sum > 0)
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / hhll[mask]) / np.log10(period)
    
    # Clip to valid range
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF HMA for trend direction
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 1d HTF Choppiness for regime detection
    chop_1d_14 = calculate_choppiness_index(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_14)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Connors RSI for entry timing
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1h HMA for short-term trend
    hma_1h_8 = calculate_hma(close, period=8)
    hma_1h_21 = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, asymmetric for crypto bias)
    POSITION_SIZE_LONG = 0.25
    POSITION_SIZE_SHORT = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Extract hour from open_time for session filter
    # open_time is in milliseconds, convert to hour
    hours = pd.to_datetime(open_time, unit='ms').hour
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(chop_1d_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1h_8[i]) or np.isnan(hma_1h_21[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === 4H TREND DIRECTION (primary filter) ===
        bull_trend_4h = close[i] > hma_4h_21_aligned[i]
        bear_trend_4h = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA slope for trend strength
        hma_4h_slope_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_slope_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # Strong trend = both price above HMA AND HMA slope aligned
        strong_bull_4h = bull_trend_4h and hma_4h_slope_bull
        strong_bear_4h = bear_trend_4h and hma_4h_slope_bear
        
        # === 1D REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range (favor mean reversion)
        # CHOP < 45 = trend (favor trend following)
        range_regime = chop_1d_aligned[i] > 55.0
        trend_regime = chop_1d_aligned[i] < 45.0
        
        # === CONNORS RSI ENTRY SIGNALS ===
        # CRSI < 15 = oversold (long entry)
        # CRSI > 85 = overbought (short entry)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # Moderate signals for trend regime
        crsi_moderate_long = crsi[i] < 35.0
        crsi_moderate_short = crsi[i] > 65.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_avg_20[i]
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during high liquidity hours
        in_session = 8 <= hours[i] <= 20
        
        # === 1H SHORT-TERM CONFIRMATION ===
        hma_1h_bull = hma_1h_8[i] > hma_1h_21[i]
        hma_1h_bear = hma_1h_8[i] < hma_1h_21[i]
        
        # === ENTRY LOGIC — STRICT CONFLUENCE FOR TRADE FREQUENCY CONTROL ===
        new_signal = 0.0
        
        # LONG ENTRIES (require 3+ confluence)
        if strong_bull_4h:
            # Condition 1: Strong 4h bull + range regime + CRSI oversold + volume
            if range_regime and crsi_oversold and volume_confirmed:
                new_signal = POSITION_SIZE_LONG
            # Condition 2: Strong 4h bull + trend regime + CRSI moderate + 1h bull + session
            elif trend_regime and crsi_moderate_long and hma_1h_bull and in_session:
                new_signal = POSITION_SIZE_LONG * 0.9
            # Condition 3: 4h bull + CRSI very oversold (<10) + volume (any regime)
            elif crsi[i] < 10.0 and volume_confirmed:
                new_signal = POSITION_SIZE_LONG * 0.8
        
        # SHORT ENTRIES (mirror logic, smaller size)
        if new_signal == 0.0 and strong_bear_4h:
            # Condition 1: Strong 4h bear + range regime + CRSI overbought + volume
            if range_regime and crsi_overbought and volume_confirmed:
                new_signal = -POSITION_SIZE_SHORT
            # Condition 2: Strong 4h bear + trend regime + CRSI moderate + 1h bear + session
            elif trend_regime and crsi_moderate_short and hma_1h_bear and in_session:
                new_signal = -POSITION_SIZE_SHORT * 0.9
            # Condition 3: 4h bear + CRSI very overbought (>90) + volume (any regime)
            elif crsi[i] > 90.0 and volume_confirmed:
                new_signal = -POSITION_SIZE_SHORT * 0.8
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
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
        
        # === EXIT CONDITIONS (regime flip or trend weakness) ===
        # Exit long on 4h trend flip to bear
        if in_position and position_side > 0:
            if bear_trend_4h and hma_4h_slope_bear:
                new_signal = 0.0
        
        # Exit short on 4h trend flip to bull
        if in_position and position_side < 0:
            if bull_trend_4h and hma_4h_slope_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals