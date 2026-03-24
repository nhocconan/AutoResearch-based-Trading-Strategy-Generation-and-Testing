#!/usr/bin/env python3
"""
Experiment #496: 30m Primary + 4h/1d HTF — Trend Pullback + Mean Reversion

Hypothesis: 30m timeframe with 4h trend bias + 30m RSI pullbacks can generate
40-80 trades/year with positive Sharpe. Use loose entry conditions (OR logic)
to ensure sufficient trade frequency while maintaining confluence.

Strategy logic:
1. 4h HMA(21) = primary trend bias (HTF filter)
2. 1d HMA(21) = higher TF confirmation (avoid counter-trend trades)
3. 30m RSI(14) extremes = pullback entries (35/65 thresholds, loose)
4. 30m Connors RSI = mean reversion confirmation (CRSI<25 long, >75 short)
5. 30m Choppiness Index = regime filter (CHOP>50 favor mean reversion)
6. 30m ATR(14)*2.5 stoploss on all positions
7. OR logic for entries to ensure trade frequency (any trigger works)

Key design choices:
- NO session filter (killed trades in #485, #489, #490, #493)
- LOOSE RSI thresholds (35/65 not 30/70) for more entries
- Multiple entry triggers (RSI + CRSI + HMA cross) with OR logic
- Size=0.25 base, 0.35 strong signals (discrete levels)
- 4h+1d HTF alignment for trend direction

Target: Sharpe>0.40, trades>=120 train (30/year), trades>=20 test
Timeframe: 30m (with 4h/1d HTF for direction)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_crsi_chop_4h1d_v1"
timeframe = "30m"
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures ranging vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hl_range = pd.Series(high - low).rolling(window=period, min_periods=period).max().values - \
               pd.Series(high - low).rolling(window=period, min_periods=period).min().values
    
    chop = np.zeros(n)
    chop[:] = np.nan
    for i in range(period, n):
        if hl_range[i] > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / hl_range[i]) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period+1):i+1]
        pos_count = np.sum(streak_vals > 0)
        if streak_period > 0:
            streak_rsi[i] = 100.0 * pos_count / streak_period
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank of returns over rank_period
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0.0], returns])
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = returns[max(0, i-rank_period+1):i+1]
        if len(window) >= rank_period:
            count_below = np.sum(window[:-1] < returns[i])
            percent_rank[i] = 100.0 * count_below / (len(window) - 1)
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 30m indicators
    hma_30m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_30m[i]) or np.isnan(rsi_14[i]):
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
        
        if np.isnan(sma_50[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h + 1d confluence) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both HTFs agree
        htf_strong_bull = htf_4h_bull and htf_1d_bull
        htf_strong_bear = htf_4h_bear and htf_1d_bear
        htf_mixed = not htf_strong_bull and not htf_strong_bear
        
        # === 30m HMA TREND ===
        hma_bull = close[i] > hma_30m[i]
        hma_bear = close[i] < hma_30m[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === RSI EXTREMES (LOOSE: 35/65 for entries) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_extreme_oversold = rsi_14[i] < 35.0
        rsi_extreme_overbought = rsi_14[i] > 65.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === CONNORS RSI (Mean Reversion) ===
        crsi_oversold = not np.isnan(crsi[i]) and crsi[i] < 25.0
        crsi_overbought = not np.isnan(crsi[i]) and crsi[i] > 75.0
        crsi_extreme_oversold = not np.isnan(crsi[i]) and crsi[i] < 15.0
        crsi_extreme_overbought = not np.isnan(crsi[i]) and crsi[i] > 85.0
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 50.0  # High chop = range market (favor mean reversion)
        chop_trend = chop[i] < 45.0  # Low chop = trending market (favor trend follow)
        
        # === HMA CROSSOVER ===
        hma_30m_prev = hma_30m[i-1] if i > 0 else hma_30m[i]
        hma_bull_cross = (close[i] > hma_30m[i]) and (close[i-1] <= hma_30m_prev) if not np.isnan(hma_30m_prev) else False
        hma_bear_cross = (close[i] < hma_30m[i]) and (close[i-1] >= hma_30m_prev) if not np.isnan(hma_30m_prev) else False
        
        # === VOLATILITY FILTER ===
        if i >= 100:
            atr_mean = np.nanmean(atr[max(0, i-100):i])
            atr_ratio = atr[i] / atr_mean if atr_mean > 1e-10 else 1.0
        else:
            atr_ratio = 1.0
        vol_normal = atr_ratio < 2.5  # Avoid entering during 2.5x normal vol
        
        # === ENTRY LOGIC (LOOSE - OR logic for trade frequency) ===
        desired_signal = 0.0
        
        # TREND LONG: HTF bull + (RSI pullback OR CRSI oversold OR HMA cross)
        if htf_strong_bull and vol_normal:
            if rsi_extreme_oversold and rsi_rising and above_sma50:
                desired_signal = SIZE_STRONG
            elif crsi_extreme_oversold and above_sma50:
                desired_signal = SIZE_STRONG
            elif hma_bull_cross and above_sma50:
                desired_signal = SIZE_BASE
            elif rsi_oversold and rsi_rising and above_sma50:
                desired_signal = SIZE_BASE
        
        # TREND SHORT: HTF bear + (RSI weakness OR CRSI overbought OR HMA cross)
        elif htf_strong_bear and vol_normal:
            if rsi_extreme_overbought and rsi_falling and below_sma50:
                desired_signal = -SIZE_STRONG
            elif crsi_extreme_overbought and below_sma50:
                desired_signal = -SIZE_STRONG
            elif hma_bear_cross and below_sma50:
                desired_signal = -SIZE_BASE
            elif rsi_overbought and rsi_falling and below_sma50:
                desired_signal = -SIZE_BASE
        
        # MEAN REVERSION LONG: CRSI extreme (works in ranging market)
        if desired_signal == 0.0 and vol_normal and chop_range:
            if crsi_extreme_oversold and above_sma200:
                desired_signal = SIZE_BASE
            elif crsi_oversold and rsi_rising and above_sma50:
                desired_signal = SIZE_BASE * 0.8
        
        # MEAN REVERSION SHORT: CRSI extreme (works in ranging market)
        if desired_signal == 0.0 and vol_normal and chop_range:
            if crsi_extreme_overbought and below_sma200:
                desired_signal = -SIZE_BASE
            elif crsi_overbought and rsi_falling and below_sma50:
                desired_signal = -SIZE_BASE * 0.8
        
        # HMA CROSSOVER (trend continuation, any HTF regime)
        if desired_signal == 0.0 and vol_normal:
            if hma_bull_cross and htf_4h_bull and above_sma50:
                desired_signal = SIZE_BASE * 0.8
            elif hma_bear_cross and htf_4h_bear and below_sma50:
                desired_signal = -SIZE_BASE * 0.8
        
        # RSI RECOVERY (simple mean reversion, works in any regime)
        if desired_signal == 0.0 and vol_normal:
            if rsi_extreme_oversold and above_sma200:
                desired_signal = SIZE_BASE * 0.8
            elif rsi_extreme_overbought and below_sma200:
                desired_signal = -SIZE_BASE * 0.8
        
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
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
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