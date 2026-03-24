#!/usr/bin/env python3
"""
Experiment #535: 6h Primary + 12h/1d HTF — Connors RSI Mean Reversion + Trend Filter

Hypothesis: Connors RSI (CRSI) provides superior mean reversion signals vs standard RSI,
especially in bear/range markets like 2022 crash and 2025 test period. CRSI combines:
1. RSI(3) - short-term momentum
2. RSI_Streak(2) - consecutive up/down day strength  
3. PercentRank(100) - where price sits in recent range

Research shows 75% win rate on CRSI extremes (<10 long, >90 short) with SMA200 filter.
By adding 12h/1d HMA trend bias, we only take mean reversion trades WITH the HTF trend,
avoiding counter-trend traps that destroyed strategies in 2022.

Key differences from failed #523/#527/#531 (6h attempts):
1. CRSI instead of standard RSI - 3-component composite is more robust
2. 12h HMA + 1d HMA dual HTF filter (not just 1d)
3. Choppiness Index regime confirmation (CHOP>55 = mean revert valid)
4. Asymmetric sizing: full size with HTF trend, half size against
5. Stricter entry: CRSI<15 (not <30) for longs, CRSI>85 for shorts

Strategy logic:
1. 1d HMA(21) = primary trend bias
2. 12h HMA(21) = secondary trend confirmation
3. 6h CRSI(3,2,100) = entry trigger (extreme mean reversion)
4. 6h Choppiness(14) = regime filter (CHOP>55 = range, mean revert OK)
5. 6h ATR(14)*2.5 = stoploss on all positions
6. 6h SMA(200) = additional trend filter (price>SMA200 for longs)

Regime-adaptive entries:
- BULL TREND (price>1d_HMA>12h_HMA): Long CRSI<15 only, no shorts
- BEAR TREND (price<1d_HMA<12h_HMA): Short CRSI>85 only, no longs
- RANGE (CHOP>60): Both directions at CRSI extremes
- TRANSITION: Half size, wait for confirmation

Target: Sharpe>0.40, trades>=60 train (15/year), trades>=8 test
Timeframe: 6h
Position Size: 0.25 base, 0.30 strong trend alignment
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_hma_chop_regime_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) - Composite mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): Standard RSI on 3-period lookback
    RSI_Streak(2): RSI of consecutive up/down streak lengths
    PercentRank(100): Where current price ranks in last 100 closes (0-100)
    
    Entry signals:
    - CRSI < 10-15: Oversold, long opportunity
    - CRSI > 85-90: Overbought, short opportunity
    """
    n = len(close)
    if n < pr_period + 1:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.zeros(n)
    rsi_short[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_short[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: Streak RSI(2)
    # Calculate streak lengths (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            if streak[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            if streak[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # Calculate RSI on streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    streak_gain = np.concatenate([[0.0], streak_gain])
    streak_loss = np.concatenate([[0.0], streak_loss])
    
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        if streak_avg_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = streak_avg_gain[i] / streak_avg_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: PercentRank(100)
    # Where does current close rank in last 100 closes?
    for i in range(pr_period, n):
        window = close[i-pr_period+1:i+1]
        rank = np.sum(window < close[i])
        pr = 100.0 * rank / pr_period
        crsi[i] = (rsi_short[i] + rsi_streak[i] + pr) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion valid)
    CHOP < 38.2 = trending (trend follow valid)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h HMA for secondary trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for primary trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_HALF = 0.15
    
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
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(sma200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (1d primary + 12h secondary) ===
        bull_trend = close[i] > hma_1d_aligned[i] and hma_1d_aligned[i] > hma_12h_aligned[i]
        bear_trend = close[i] < hma_1d_aligned[i] and hma_1d_aligned[i] < hma_12h_aligned[i]
        neutral_trend = not bull_trend and not bear_trend
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma200[i]
        below_sma200 = close[i] < sma200[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0   # Range-bound (mean revert OK)
        chop_trend = chop[i] < 45.0   # Trending (trend follow OK)
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0    # Long entry zone
        crsi_overbought = crsi[i] > 85.0  # Short entry zone
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === CRSI RECOVERY (exiting extreme) ===
        crsi_recovering_long = crsi_oversold and i > 0 and crsi[i] > crsi[i-1]
        crsi_recovering_short = crsi_overbought and i > 0 and crsi[i] < crsi[i-1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # BULL TREND REGIME: Only long entries on CRSI oversold
        if bull_trend and above_sma200:
            if crsi_extreme_oversold:
                desired_signal = SIZE_STRONG  # Full size, strong confluence
            elif crsi_oversold and chop_range:
                desired_signal = SIZE_BASE    # Base size with range confirmation
            elif crsi_recovering_long:
                desired_signal = SIZE_BASE    # Recovery entry
        
        # BEAR TREND REGIME: Only short entries on CRSI overbought
        elif bear_trend and below_sma200:
            if crsi_extreme_overbought:
                desired_signal = -SIZE_STRONG  # Full size, strong confluence
            elif crsi_overbought and chop_range:
                desired_signal = -SIZE_BASE    # Base size with range confirmation
            elif crsi_recovering_short:
                desired_signal = -SIZE_BASE    # Recovery entry
        
        # RANGE/NEUTRAL REGIME: Both directions allowed at extremes
        elif neutral_trend or chop_range:
            if crsi_extreme_oversold and above_sma200:
                desired_signal = SIZE_BASE     # Half-normal size in range
            elif crsi_extreme_overbought and below_sma200:
                desired_signal = -SIZE_BASE    # Half-normal size in range
            elif crsi_oversold and crsi_recovering_long:
                desired_signal = SIZE_HALF     # Reduced size
            elif crsi_overbought and crsi_recovering_short:
                desired_signal = -SIZE_HALF    # Reduced size
        
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
        elif desired_signal >= SIZE_HALF * 0.9:
            final_signal = SIZE_HALF
        elif desired_signal <= -SIZE_HALF * 0.9:
            final_signal = -SIZE_HALF
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