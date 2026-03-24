#!/usr/bin/env python3
"""
Experiment #611: 6h Primary + 1d/1w HTF — Asymmetric Regime + Connors RSI Pullback

Hypothesis: 6h timeframe captures multi-day swings better than 4h (less noise) and 12h (more trades).
Key insight from 200+ failed experiments: OVER-FILTERING causes 0 trades. Simple regime detection
with asymmetric entry logic (bull=long only, bear=short only) produces more consistent signals.

Why this differs from failed #600/603/607:
1. SIMPLER regime: price vs 1d HMA only (not ADX+CHOP+multiple HTF)
2. ASYMMETRIC entries: bull regime = long pullbacks ONLY, bear regime = short rallies ONLY
3. CONNORS RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 - proven 75% win rate
4. VOLUME confirmation: entry volume > 1.2x 20-bar average (avoids fake breakouts)
5. FEWER filters: HTF sets regime direction, not entry gate (prevents 0 trades)

Strategy logic:
1. 1w HMA(21) = macro bias (only used for regime strength, not entry filter)
2. 1d HMA(21) = primary regime filter (price above = bull, below = bear)
3. 6h HMA(16) = trend direction for entry timing
4. 6h Connors RSI = pullback entry trigger (CRSI<15 long, CRSI>85 short)
5. 6h Volume = confirmation (vol > 1.2x avg = valid move)
6. ATR(14)*2.5 stoploss on all positions

Regime-adaptive entries:
- BULL (price > 1d HMA): Long only on CRSI<15 pullback + volume confirm
- BEAR (price < 1d HMA): Short only on CRSI>85 rally + volume confirm
- TRANSITION (price near 1d HMA): Flat or reduced size

Target: Sharpe>0.45, trades>=100 train (25/year), trades>=12 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_asymmetric_crsi_vol_1d1w_v1"
timeframe = "6h"
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

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak Component for Connors RSI
    Counts consecutive up/down days, converts to RSI
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    streak = np.zeros(n)
    streak[0] = 0
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        streak_values = streak[max(0, i-period+1):i+1]
        up_streaks = np.sum(streak_values > 0)
        if period > 0:
            streak_rsi[i] = 100.0 * up_streaks / period
        else:
            streak_rsi[i] = 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank Component for Connors RSI
    Where does current return rank vs last N periods?
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    returns = np.zeros(n)
    returns[0] = 0
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
        else:
            returns[i] = 0.0
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current)
        percent_rank[i] = 100.0 * rank / period
    
    return percent_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Extreme values (<10 or >90) indicate reversal opportunities
    Proven 75% win rate on mean reversion entries
    """
    rsi_short = calculate_rsi(close, period=rsi_period)
    rsi_streak = calculate_rsi_streak(close, period=streak_period)
    percent_rk = calculate_percent_rank(close, period=pr_period)
    
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rk[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + percent_rk[i]) / 3.0
    
    return crsi

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

def calculate_volume_ma(volume, period=20):
    """Volume moving average for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for primary regime filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro bias strength
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    hma_6h = calculate_hma(close, period=16)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BULL = 0.30
    SIZE_BEAR = 0.30
    SIZE_REDUCE = 0.15
    
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
        
        if np.isnan(hma_6h[i]) or np.isnan(crsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (1d HMA) ===
        bull_regime = close[i] > hma_1d_aligned[i]
        bear_regime = close[i] < hma_1d_aligned[i]
        
        # === MACRO STRENGTH (1w HMA) ===
        # Strong bull: price > 1d HMA AND 1d HMA > 1w HMA
        # Strong bear: price < 1d HMA AND 1d HMA < 1w HMA
        macro_bull = hma_1d_aligned[i] > hma_1w_aligned[i]
        macro_bear = hma_1d_aligned[i] < hma_1w_aligned[i]
        
        # === 6h TREND (HMA slope) ===
        hma_slope_bull = hma_6h[i] > hma_6h[i-10] if i >= 10 and not np.isnan(hma_6h[i-10]) else False
        hma_slope_bear = hma_6h[i] < hma_6h[i-10] if i >= 10 and not np.isnan(hma_6h[i-10]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = volume[i] > 1.2 * vol_ma[i] if vol_ma[i] > 1e-10 else False
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === ASYMMETRIC ENTRY LOGIC ===
        desired_signal = 0.0
        
        # BULL REGIME: Long only on pullbacks (CRSI oversold)
        if bull_regime:
            # Strong bull + extreme oversold = full size
            if macro_bull and crsi_extreme_oversold and hma_slope_bull:
                desired_signal = SIZE_BULL
            # Bull regime + oversold pullback = standard size
            elif crsi_oversold and hma_slope_bull:
                desired_signal = SIZE_BULL
            # Bull regime + moderate oversold = reduced size
            elif crsi[i] < 25.0 and close[i] > hma_6h[i]:
                desired_signal = SIZE_REDUCE
        
        # BEAR REGIME: Short only on rallies (CRSI overbought)
        elif bear_regime:
            # Strong bear + extreme overbought = full size
            if macro_bear and crsi_extreme_overbought and hma_slope_bear:
                desired_signal = -SIZE_BEAR
            # Bear regime + overbought rally = standard size
            elif crsi_overbought and hma_slope_bear:
                desired_signal = -SIZE_BEAR
            # Bear regime + moderate overbought = reduced size
            elif crsi[i] > 75.0 and close[i] < hma_6h[i]:
                desired_signal = -SIZE_REDUCE
        
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
        if desired_signal >= SIZE_BULL * 0.9:
            final_signal = SIZE_BULL
        elif desired_signal <= -SIZE_BEAR * 0.9:
            final_signal = -SIZE_BEAR
        elif desired_signal >= SIZE_REDUCE * 0.9:
            final_signal = SIZE_REDUCE
        elif desired_signal <= -SIZE_REDUCE * 0.9:
            final_signal = -SIZE_REDUCE
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