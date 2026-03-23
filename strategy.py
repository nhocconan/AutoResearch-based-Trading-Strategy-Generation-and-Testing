#!/usr/bin/env python3
"""
Experiment #1237: 1d Primary + 1w HTF — Connors RSI Mean Reversion with Weekly Trend Filter

Hypothesis: Daily timeframe with weekly trend filter reduces noise while maintaining
trade frequency. Connors RSI (CRSI) has proven 75%+ win rate for mean reversion when
combined with trend filter. Research shows CRSI works best on 1d timeframe because:
1) Daily bars filter intraday noise
2) Weekly HMA provides clear macro bias
3) Mean reversion works better in daily timeframes than lower TFs

Key components:
- Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- 1w HMA for macro trend direction (from mtf_data helper)
- ADX filter to avoid trading in extreme chop (ADX < 18 = skip)
- ATR trailing stoploss (2.5x)
- Discrete signal sizes: 0.0, ±0.28

Target: Sharpe > 0.612, trades >= 30 train (7.5/year), >= 3 test, DD > -50%
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_meanrevert_1w_hma_adx_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - Composite mean reversion indicator
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) on close - short-term momentum
    2. RSI(2) on streak - consecutive up/down days
    3. PercentRank(100) - where close ranks in last 100 days
    
    Extreme readings (<10 or >90) signal mean reversion opportunities
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # Component 1: RSI(3) on close
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_3 = pd.Series(gain).ewm(span=3, min_periods=3, adjust=False).mean().values
    loss_3 = pd.Series(loss).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    rs_3 = np.zeros(n)
    mask_3 = loss_3 > 1e-10
    rs_3[mask_3] = gain_3[mask_3] / loss_3[mask_3]
    rsi_3 = 100.0 - (100.0 / (1.0 + rs_3))
    rsi_3[:3] = np.nan
    
    # Component 2: RSI(2) on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_gain = np.zeros(n)
    streak_loss = np.zeros(n)
    streak_gain[1:] = np.where(streak[1:] > 0, streak[1:], 0)
    streak_loss[1:] = np.where(streak[1:] < 0, -streak[1:], 0)
    
    streak_gain_2 = pd.Series(streak_gain).ewm(span=2, min_periods=2, adjust=False).mean().values
    streak_loss_2 = pd.Series(streak_loss).ewm(span=2, min_periods=2, adjust=False).mean().values
    
    rs_streak = np.zeros(n)
    mask_streak = streak_loss_2 > 1e-10
    rs_streak[mask_streak] = streak_gain_2[mask_streak] / streak_loss_2[mask_streak]
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak[:2] = np.nan
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength filter"""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    di_sum = plus_di + minus_di
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss"""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Also calculate 1d HMA for additional trend confirmation
    hma_1d = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(adx[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d[i]):
            continue
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === LOCAL TREND (1d HMA) ===
        local_bull = close[i] > hma_1d[i]
        local_bear = close[i] < hma_1d[i]
        
        # === TREND STRENGTH FILTER (ADX) ===
        # Skip trading when market is too choppy (ADX < 15)
        tradeable = adx[i] >= 15.0
        
        # === CONNORS RSI SIGNALS ===
        # Long: CRSI extremely oversold (< 12) + macro bull trend
        # Short: CRSI extremely overbought (> 88) + macro bear trend
        crsi_oversold = crsi[i] < 12.0
        crsi_overbought = crsi[i] > 88.0
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        if not tradeable:
            desired_signal = 0.0
        else:
            # LONG: Macro bull + local bull + CRSI oversold (mean reversion in uptrend)
            if macro_bull and local_bull and crsi_oversold:
                desired_signal = BASE_SIZE
            
            # SHORT: Macro bear + local bear + CRSI overbought (mean reversion in downtrend)
            elif macro_bear and local_bear and crsi_overbought:
                desired_signal = -BASE_SIZE
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals