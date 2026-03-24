#!/usr/bin/env python3
"""
Experiment #565: 15m Primary + 4h/1d HTF — Session RSI Pullback + CPR Levels

Hypothesis: 15m timeframe is underexplored (0 experiments). Key insight: 15m needs 
VERY selective entries to avoid fee drag. This strategy uses:
1. 1d Central Pivot Range (CPR) for key support/resistance levels
2. 4h HMA(21) for trend direction bias
3. 15m RSI(7) for fast entry timing (Connors-style short RSI)
4. Session filter: prefer 00-12 UTC (London/NY overlap = higher volume)
5. Volatility filter: ATR(14)/ATR(50) ratio > 1.2 (only trade when vol expanding)
6. 3+ confluence required: HTF trend + session + RSI extreme + vol filter

Why this might work on 15m:
- HTF (4h/1d) provides direction → reduces whipsaw
- Short RSI(7) catches intraday pullbacks faster than RSI(14)
- Session filter avoids low-volume Asian session chop
- Vol filter ensures we trade during momentum, not dead periods
- Discrete sizing (0.20/0.25) minimizes fee churn

Target: 50-80 trades/year, Sharpe>0.40, DD<-30%
Timeframe: 15m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_rsi_cpr_4h1d_v1"
timeframe = "15m"
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

def calculate_cpr_levels(open_price, high, low, close):
    """
    Central Pivot Range (CPR) - Daily pivot levels
    Pivot = (High + Low + Close) / 3
    BC (Bottom Central) = (High + Low) / 2
    TC (Top Central) = Pivot
    Narrow CPR = TC - BC < threshold (indicates potential breakout day)
    """
    n = len(close)
    pivot = np.zeros(n)
    bc = np.zeros(n)
    tc = np.zeros(n)
    
    for i in range(n):
        pivot[i] = (high[i] + low[i] + close[i]) / 3.0
        bc[i] = (high[i] + low[i]) / 2.0
        tc[i] = pivot[i]
    
    return pivot, bc, tc

def calculate_streak_rsi(close, period=2):
    """
    Connors RSI component: RSI of up/down streaks
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    streak = np.zeros(n)
    direction = np.zeros(n)  # 1 = up, -1 = down, 0 = neutral
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            direction[i] = 1
            if direction[i-1] == 1:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            direction[i] = -1
            if direction[i-1] == -1:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = -1
        else:
            direction[i] = 0
            streak[i] = 0
    
    # RSI of absolute streak values
    abs_streak = np.abs(streak)
    streak_rsi = calculate_rsi(abs_streak, period)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Connors RSI component: Percentile Rank of current close vs last N periods
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        count_below = np.sum(window < close[i])
        pr[i] = 100.0 * count_below / period
    
    return pr

def calculate_conners_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Extreme oversold: CRSI < 10
    Extreme overbought: CRSI > 90
    """
    n = len(close)
    if n < pr_period + 5:
        return np.full(n, np.nan)
    
    rsi_close = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_streak_rsi(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(pr_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + pr[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_price = prices["open"].values
    n = len(close)
    
    # Extract hour from open_time for session filter
    # open_time is in milliseconds since epoch
    hours = (prices["open_time"].values // (1000 * 60 * 60)) % 24
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d CPR levels
    pivot_1d_raw, bc_1d_raw, tc_1d_raw = calculate_cpr_levels(
        df_1d['open'].values, df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_raw)
    bc_1d_aligned = align_htf_to_ltf(prices, df_1d, bc_1d_raw)
    tc_1d_aligned = align_htf_to_ltf(prices, df_1d, tc_1d_raw)
    
    # Calculate 1d HMA for additional trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)  # Fast RSI for entries
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_50 = calculate_atr(high, low, close, period=50)
    
    # Connors RSI for mean reversion signals
    crsi = calculate_conners_rsi(close, rsi_period=3, streak_period=2, pr_period=50)
    
    # Volatility ratio (ATR expansion filter)
    vol_ratio = np.zeros(n)
    vol_ratio[:] = np.nan
    for i in range(50, n):
        if atr_50[i] > 1e-10 and not np.isnan(atr_14[i]):
            vol_ratio[i] = atr_14[i] / atr_50[i]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(bc_1d_aligned[i]) or np.isnan(tc_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h + 1d) ===
        htf_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_1d_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === 4h HMA SLOPE ===
        hma_4h_slope_bull = False
        hma_4h_slope_bear = False
        if i >= 10 and not np.isnan(hma_4h_aligned[i-10]):
            hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-10]
            hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-10]
        
        # === CPR POSITION (price relative to daily pivot) ===
        # Narrow CPR = potential breakout day
        cpr_width = abs(tc_1d_aligned[i] - bc_1d_aligned[i])
        avg_cpr_width = np.nanmean(np.abs(tc_1d_aligned[max(0,i-20):i+1] - bc_1d_aligned[max(0,i-20):i+1]))
        narrow_cpr = cpr_width < 0.5 * avg_cpr_width if avg_cpr_width > 1e-10 else False
        
        # Price above/below CPR
        price_above_cpr = close[i] > tc_1d_aligned[i]
        price_below_cpr = close[i] < bc_1d_aligned[i]
        price_in_cpr = bc_1d_aligned[i] <= close[i] <= tc_1d_aligned[i]
        
        # === SESSION FILTER (00-12 UTC = London/NY overlap) ===
        is_good_session = 0 <= hours[i] <= 12
        
        # === VOLATILITY FILTER (ATR expansion) ===
        vol_expanding = vol_ratio[i] > 1.15  # ATR(14) > 115% of ATR(50)
        
        # === RSI SIGNALS (15m) ===
        rsi_7_oversold = rsi_7[i] < 35.0
        rsi_7_overbought = rsi_7[i] > 65.0
        rsi_7_extreme_oversold = rsi_7[i] < 25.0
        rsi_7_extreme_overbought = rsi_7[i] > 75.0
        
        # RSI recovery
        rsi_7_rising = rsi_7[i] > rsi_7[i-1] if i > 0 and not np.isnan(rsi_7[i-1]) else False
        rsi_7_falling = rsi_7[i] < rsi_7[i-1] if i > 0 and not np.isnan(rsi_7[i-1]) else False
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = not np.isnan(crsi[i]) and crsi[i] < 15.0
        crsi_overbought = not np.isnan(crsi[i]) and crsi[i] > 85.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        confluence_count = 0
        
        # LONG entries (need 3+ confluence)
        if htf_bull or (htf_neutral and hma_4h_slope_bull):
            confluence = 0
            
            # Confluence 1: HTF bullish bias
            if htf_bull:
                confluence += 1
            
            # Confluence 2: RSI oversold bounce
            if rsi_7_oversold and rsi_7_rising:
                confluence += 1
            
            # Confluence 3: Good session
            if is_good_session:
                confluence += 1
            
            # Confluence 4: Volatility expanding
            if vol_expanding:
                confluence += 1
            
            # Confluence 5: Price near support (CPR or pullback)
            if price_in_cpr or price_below_cpr:
                confluence += 1
            
            # Confluence 6: Connors RSI extreme
            if crsi_oversold:
                confluence += 1
            
            if confluence >= 3:
                if rsi_7_extreme_oversold or crsi_oversold:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            confluence_count = max(confluence_count, confluence)
        
        # SHORT entries (need 3+ confluence)
        if htf_bear or (htf_neutral and hma_4h_slope_bear):
            confluence = 0
            
            # Confluence 1: HTF bearish bias
            if htf_bear:
                confluence += 1
            
            # Confluence 2: RSI overbought fade
            if rsi_7_overbought and rsi_7_falling:
                confluence += 1
            
            # Confluence 3: Good session
            if is_good_session:
                confluence += 1
            
            # Confluence 4: Volatility expanding
            if vol_expanding:
                confluence += 1
            
            # Confluence 5: Price near resistance (CPR or rally)
            if price_in_cpr or price_above_cpr:
                confluence += 1
            
            # Confluence 6: Connors RSI extreme
            if crsi_overbought:
                confluence += 1
            
            if confluence >= 3:
                if rsi_7_extreme_overbought or crsi_overbought:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            
            confluence_count = max(confluence_count, confluence)
        
        # === STOPLOSS CHECK (2x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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