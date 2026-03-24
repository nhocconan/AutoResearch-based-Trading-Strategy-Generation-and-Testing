#!/usr/bin/env python3
"""
Experiment #166: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: Daily timeframe with weekly trend filter offers optimal trade frequency
(20-50/year) with minimal fee drag. Choppiness Index detects regime (trend vs range),
Connors RSI provides sensitive entry signals, and 1w HMA ensures we trade with
major trend direction.

Key design choices:
- 1w HMA(50) for weekly trend bias (only long when price > 1w HMA, only short when <)
- Choppiness Index(14) for regime: >55 = range (mean revert), <45 = trend (breakout)
- Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
  - More sensitive than regular RSI, catches reversals faster
  - Long when CRSI < 15, Short when CRSI > 85
- Donchian(20) breakout confirmation for trend regime
- ATR(14) trailing stop at 2.5x for risk management
- Position size: 0.30 (30% of capital)

Trade generation strategy (CRITICAL - avoid 0 trades):
- LOOSE CRSI thresholds (15/85 not 10/90)
- Dual regime logic ensures entries in both trending AND ranging markets
- Fallback entries when weekly trend is very strong (ignore chop filter)
- Target 25-40 trades/year on 1d timeframe

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_hma_1w_v2"
timeframe = "1d"
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
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    More sensitive than regular RSI, designed for short-term mean reversion
    Long when CRSI < 10-15, Short when CRSI > 85-90
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - measure consecutive up/down days
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(streak_period, n):
        streak = 0
        for j in range(i, max(0, i - 20), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                streak += 1
            elif close[j] < close[j-1]:
                streak -= 1
            else:
                break
        
        # Convert streak to RSI-like value
        if streak >= 0:
            streak_rsi[i] = min(100.0, streak * 25.0)
        else:
            streak_rsi[i] = max(0.0, 100.0 + streak * 25.0)
    
    # Percent Rank - where current price ranks in last N periods
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_donchian_high(high, period=20):
    """Donchian Channel Upper Band (highest high of last N periods)"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    donchian = np.zeros(n)
    donchian[:] = np.nan
    
    for i in range(period - 1, n):
        donchian[i] = high[i-period+1:i+1].max()
    
    return donchian

def calculate_donchian_low(low, period=20):
    """Donchian Channel Lower Band (lowest low of last N periods)"""
    n = len(low)
    if n < period:
        return np.full(n, np.nan)
    
    donchian = np.zeros(n)
    donchian[:] = np.nan
    
    for i in range(period - 1, n):
        donchian[i] = low[i-period+1:i+1].min()
    
    return donchian

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for weekly regime filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_1d = calculate_hma(close, period=21)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_high = calculate_donchian_high(high, period=20)
    donchian_low = calculate_donchian_low(low, period=20)
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after 200 SMA is ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WEEKLY REGIME (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = ranging (use mean reversion)
        # CHOP < 45 = trending (use breakout)
        is_range = chop[i] > 55.0
        is_trend = chop[i] < 45.0
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Mean reversion long
        crsi_overbought = crsi[i] > 85.0  # Mean reversion short
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_high[i-1] if not np.isnan(donchian_high[i-1]) else False
        breakout_short = close[i] < donchian_low[i-1] if not np.isnan(donchian_low[i-1]) else False
        
        # === HMA TREND ===
        hma_bull = close[i] > hma_1d[i]
        hma_bear = close[i] < hma_1d[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING MARKET (CHOP < 45) - Follow breakout with HTF
        if is_trend:
            # Long: Weekly bull + Donchian breakout + HMA bull + above SMA200
            if htf_1w_bull and breakout_long and hma_bull and above_sma200:
                desired_signal = SIZE
            
            # Short: Weekly bear + Donchian breakdown + HMA bear + below SMA200
            elif htf_1w_bear and breakout_short and hma_bear and below_sma200:
                desired_signal = -SIZE
        
        # REGIME 2: RANGING MARKET (CHOP > 55) - Mean reversion with HTF
        elif is_range:
            # Long: Weekly bull + CRSI oversold + above SMA200
            if htf_1w_bull and crsi_oversold and above_sma200:
                desired_signal = SIZE
            
            # Short: Weekly bear + CRSI overbought + below SMA200
            elif htf_1w_bear and crsi_overbought and below_sma200:
                desired_signal = -SIZE
        
        # FALLBACK 1: Strong weekly trend (ignore chop filter) - 80% size
        # Ensures trades when weekly momentum is very strong
        if desired_signal == 0.0:
            if htf_1w_bull and hma_bull and crsi[i] < 30.0 and above_sma200:
                desired_signal = SIZE * 0.8
            elif htf_1w_bear and hma_bear and crsi[i] > 70.0 and below_sma200:
                desired_signal = -SIZE * 0.8
        
        # FALLBACK 2: CRSI extreme (ignore weekly, use SMA200) - 60% size
        # Ensures trades during strong mean reversion setups
        if desired_signal == 0.0:
            if crsi_oversold and above_sma200 and hma_bull:
                desired_signal = SIZE * 0.6
            elif crsi_overbought and below_sma200 and hma_bear:
                desired_signal = -SIZE * 0.6
        
        # FALLBACK 3: Donchian breakout with HMA confirmation - 50% size
        # Catches strong momentum moves
        if desired_signal == 0.0:
            if breakout_long and hma_bull and above_sma200:
                desired_signal = SIZE * 0.5
            elif breakout_short and hma_bear and below_sma200:
                desired_signal = -SIZE * 0.5
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.7:
            final_signal = SIZE * 0.8
        elif desired_signal <= -SIZE * 0.7:
            final_signal = -SIZE * 0.8
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.6
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.6
        elif desired_signal >= SIZE * 0.3:
            final_signal = SIZE * 0.4
        elif desired_signal <= -SIZE * 0.3:
            final_signal = -SIZE * 0.4
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals