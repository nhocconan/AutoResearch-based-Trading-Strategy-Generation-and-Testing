#!/usr/bin/env python3
"""
Experiment #664: 12h Primary + 1d/1w HTF — Dual Regime Adaptive (Choppiness + Connors RSI)

Hypothesis: 12h timeframe with regime-adaptive logic should excel in mixed 2021-2025 markets.
Uses Choppiness Index to detect range vs trend, then switches strategy:
- CHOP > 55 (range): Connors RSI mean reversion at extremes
- CHOP < 45 (trend): HMA breakout with HTF bias

Key innovations:
1. Choppiness Index(14) regime filter - switches between mean-revert and trend-follow
2. Connors RSI for mean-reversion entries (RSI3 + StreakRSI2 + PercentRank100) / 3
3. 1d HMA(21) + 1w HMA(21) dual HTF bias - only trade in direction of both
4. Asymmetric entries - long bias in bull HTF, short bias in bear HTF
5. ATR(14) trailing stop - 2.5x for risk management
6. Size: 0.25 base, 0.30 strong signals (discrete to minimize churn)

Entry conditions (LOOSE to ensure trades):
- RANGE mode (CHOP>55): CRSI<15 long, CRSI>85 short + HTF bias alignment
- TREND mode (CHOP<45): Price crosses HMA(21) + HTF bias + ADX>18
- TRANSITION (45-55): Half size, either signal valid

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 12h (20-50 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_crsi_chop_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh == ll:
            chop[i] = 100.0
            continue
        
        # Sum of True Range over period
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        if tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(hh - ll) - np.log10(tr_sum)
            chop[i] = 100.0 * (np.log10(hh - ll) - np.log10(tr_sum)) / np.log10(period)
        else:
            chop[i] = 100.0
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Extreme readings (<10 or >90) indicate mean-reversion opportunities
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi3 = np.zeros(n)
    rsi3[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi3[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi3[i] = 100.0
    
    # Streak RSI(2) - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        # Calculate average streak over period
        if i >= streak_period:
            up_streaks = 0
            down_streaks = 0
            for j in range(i-streak_period+1, i+1):
                if streak[j] > 0:
                    up_streaks += streak[j]
                elif streak[j] < 0:
                    down_streaks += abs(streak[j])
            
            if up_streaks + down_streaks > 1e-10:
                streak_rsi[i] = 100.0 * up_streaks / (up_streaks + down_streaks)
            else:
                streak_rsi[i] = 50.0
    
    # Percent Rank(100) - where does current return rank vs last 100 bars
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(rank_period, n):
        current_return = (close[i] - close[i-1]) / close[i-1] if close[i-1] > 1e-10 else 0
        returns_window = []
        for j in range(i-rank_period+1, i):
            ret = (close[j] - close[j-1]) / close[j-1] if close[j-1] > 1e-10 else 0
            returns_window.append(ret)
        
        if len(returns_window) > 0:
            count_below = sum(1 for r in returns_window if r < current_return)
            pct_rank[i] = 100.0 * count_below / len(returns_window)
        else:
            pct_rank[i] = 50.0
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi3[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0.0)
        else:
            plus_dm[i] = 0.0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0.0)
        else:
            minus_dm[i] = 0.0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_21 = calculate_hma(close, period=21)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(hma_21[i]):
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
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong HTF bias only when both agree
        htf_strong_bull = htf_1d_bull and htf_1w_bull
        htf_strong_bear = htf_1d_bear and htf_1w_bear
        htf_neutral = not htf_strong_bull and not htf_strong_bear
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_range = chop[i] > 55.0  # Mean reversion mode
        regime_trend = chop[i] < 45.0  # Trend following mode
        regime_transition = not regime_range and not regime_trend
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # RANGE MODE: Connors RSI mean reversion
        if regime_range:
            # Long: CRSI oversold + HTF not strongly bearish
            if crsi[i] < 20 and not htf_strong_bear:
                if htf_strong_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            # Short: CRSI overbought + HTF not strongly bullish
            elif crsi[i] > 80 and not htf_strong_bull:
                if htf_strong_bear:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # TREND MODE: HMA breakout with ADX confirmation
        elif regime_trend:
            hma_bull = close[i] > hma_21[i] and (i < 2 or hma_21[i] > hma_21[i-1])
            hma_bear = close[i] < hma_21[i] and (i < 2 or hma_21[i] < hma_21[i-1])
            trend_confirmed = adx[i] > 18.0
            
            # Long: Price above HMA + HMA rising + ADX confirms + HTF bull
            if hma_bull and trend_confirmed and htf_strong_bull:
                desired_signal = SIZE_STRONG
            elif hma_bull and trend_confirmed:
                desired_signal = SIZE_BASE
            
            # Short: Price below HMA + HMA falling + ADX confirms + HTF bear
            elif hma_bear and trend_confirmed and htf_strong_bear:
                desired_signal = -SIZE_STRONG
            elif hma_bear and trend_confirmed:
                desired_signal = -SIZE_BASE
        
        # TRANSITION MODE: Half size, either signal valid
        elif regime_transition:
            # Mean reversion signals (looser thresholds)
            if crsi[i] < 25 and not htf_strong_bear:
                desired_signal = SIZE_BASE * 0.5
            elif crsi[i] > 75 and not htf_strong_bull:
                desired_signal = -SIZE_BASE * 0.5
            # Trend signals (looser ADX)
            elif close[i] > hma_21[i] and htf_strong_bull:
                desired_signal = SIZE_BASE * 0.5
            elif close[i] < hma_21[i] and htf_strong_bear:
                desired_signal = -SIZE_BASE * 0.5
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
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