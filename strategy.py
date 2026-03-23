#!/usr/bin/env python3
"""
Experiment #1111: 4h Primary + 1d/1w HTF — Dual Regime Strategy

Hypothesis: After 806+ failed experiments, the key insight is:
1. Single-regime strategies fail because markets switch between trend/range
2. Choppiness Index (CHOP) reliably detects regime: CHOP>55=range, CHOP<45=trend
3. In RANGE regime: Use Connors RSI mean reversion (buy oversold, sell overbought)
4. In TREND regime: Use HMA trend + RSI pullback entries
5. 1w HMA for ultra-macro filter, 1d HMA for intermediate trend
6. Looser thresholds ensure adequate trade frequency (RSI 35/65, ADX 18)
7. Position size 0.30 base with 2.5x ATR trailing stop

Why this should beat Sharpe=0.612:
- Dual regime adapts to market conditions (works in 2022 crash AND 2021 bull)
- Connors RSI proven in research (ETH Sharpe +0.923)
- Choppiness filter prevents trend strategies from dying in chop
- 4h timeframe naturally gives 20-50 trades/year
- 1w HTF provides strong macro bias without over-trading

Timeframe: 4h (primary)
HTF: 1d, 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.30 base, 0.15 reduced (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 30-60 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_crsi_hma_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    sqrt_period = max(1, int(np.sqrt(period)))
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — composite mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI_Streak(2): RSI of consecutive up/down days
    3. PercentRank(100): Where current price ranks vs last 100 bars
    
    Entry: CRSI < 10 (oversold) or CRSI > 90 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_rsi = np.full(n, 50.0)
    for i in range(streak_period, n):
        avg_gain = np.mean(streak_gain[max(0, i-streak_period+1):i+1])
        avg_loss = np.mean(streak_loss[max(0, i-streak_period+1):i+1])
        if avg_loss > 1e-10:
            rs = avg_gain / avg_loss
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            streak_rsi[i] = 100.0 if avg_gain > 0 else 50.0
    
    # Component 3: Percent Rank
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine components
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index — measures if market is trending or ranging.
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = Range/Chop (mean reversion works)
    - CHOP < 38.2 = Trend (trend following works)
    - 38.2 < CHOP < 61.8 = Transition
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR and High/Low range
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — measures trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = tr_s > 1e-10
    plus_di[mask] = 100.0 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100.0 * minus_dm_s[mask] / tr_s[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    crsi_4h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_4h = calculate_choppiness_index(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    # HMA for trend direction on 4h
    hma_4h_fast = calculate_hma(close, period=16)
    hma_4h_slow = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr[i]) or np.isnan(adx[i]):
            continue
        if np.isnan(chop_4h[i]) or np.isnan(crsi_4h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(hma_4h_fast[i]) or np.isnan(hma_4h_slow[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = Range (mean reversion works)
        # CHOP < 45 = Trend (trend following works)
        is_range_regime = chop_4h[i] > 55.0
        is_trend_regime = chop_4h[i] < 45.0
        
        # === MACRO TREND FILTERS ===
        # 1w HMA = ultra-macro bias
        # 1d HMA = intermediate trend
        macro_bull_1w = close[i] > hma_1w_aligned[i]
        macro_bear_1w = close[i] < hma_1w_aligned[i]
        macro_bull_1d = close[i] > hma_1d_aligned[i]
        macro_bear_1d = close[i] < hma_1d_aligned[i]
        
        # 4h HMA crossover for trend direction
        hma_bull = hma_4h_fast[i] > hma_4h_slow[i]
        hma_bear = hma_4h_fast[i] < hma_4h_slow[i]
        
        # === RANGE REGIME: Mean Reversion with Connors RSI ===
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        if is_range_regime:
            # Connors RSI extremes for mean reversion
            crsi_oversold = crsi_4h[i] < 25.0
            crsi_overbought = crsi_4h[i] > 75.0
            
            # Long: CRSI oversold + macro not strongly bearish
            if crsi_oversold and not macro_bear_1w:
                desired_signal = current_size
            
            # Short: CRSI overbought + macro not strongly bullish
            elif crsi_overbought and not macro_bull_1w:
                desired_signal = -current_size
        
        # === TREND REGIME: HMA Trend + RSI Pullback ===
        elif is_trend_regime:
            # ADX filter for trend strength (loose threshold)
            trend_strong = adx[i] > 18.0
            
            # RSI pullback levels (loose for more trades)
            rsi_pullback_long = rsi_4h[i] < 50.0 and rsi_4h[i] > 35.0
            rsi_pullback_short = rsi_4h[i] > 50.0 and rsi_4h[i] < 65.0
            
            # Long: HMA bull + ADX strong + RSI pullback + 1d macro ok
            if hma_bull and trend_strong and rsi_pullback_long and not macro_bear_1d:
                desired_signal = current_size
            
            # Short: HMA bear + ADX strong + RSI pullback + 1d macro ok
            elif hma_bear and trend_strong and rsi_pullback_short and not macro_bull_1d:
                desired_signal = -current_size
        
        # === TRANSITION REGIME (45 < CHOP < 55): Reduced position or flat ===
        else:
            # Only take high-conviction trades in transition
            if macro_bull_1w and macro_bull_1d and rsi_4h[i] < 40.0:
                desired_signal = REDUCED_SIZE
            elif macro_bear_1w and macro_bear_1d and rsi_4h[i] > 60.0:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if range regime with CRSI not overbought OR trend regime intact
                if (is_range_regime and crsi_4h[i] < 70.0) or (is_trend_regime and hma_bull):
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if range regime with CRSI not oversold OR trend regime intact
                if (is_range_regime and crsi_4h[i] > 30.0) or (is_trend_regime and hma_bear):
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if regime changes or CRSI overbought in range
            if (is_range_regime and crsi_4h[i] > 80.0) or (is_trend_regime and hma_bear):
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if regime changes or CRSI oversold in range
            if (is_range_regime and crsi_4h[i] < 20.0) or (is_trend_regime and hma_bull):
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
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