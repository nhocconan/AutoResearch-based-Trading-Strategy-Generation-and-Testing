#!/usr/bin/env python3
"""
Experiment #866: 12h Primary + 1d HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 602+ failed strategies, the key insight is:
1. 12h timeframe needs MEAN REVERSION entries (not pure trend) for bear/range markets
2. Connors RSI has 75% win rate and works through 2022 crash and 2025 bear
3. Choppiness Index regime filter prevents trend strategies in chop (major loss source)
4. 1d HMA(21) for long-term bias only — not entry trigger
5. CRITICAL: Relaxed entry thresholds to ensure MINIMUM 30 trades/symbol (0-trade = auto-reject)
6. Multiple entry pathways — don't require ALL conditions to agree

Strategy design:
1. 12h Primary timeframe (target 25-50 trades/year)
2. 1d HMA(21) for long-term bias (aligned via mtf_data helper)
3. Connors RSI(2,2,100) for entry timing — proven mean reversion edge
4. Choppiness Index(14) for regime detection — CHOP>55=range, CHOP<45=trend
5. 12h RSI(14) for confluence
6. 12h ATR(14) for trailing stop (2.5x)
7. Dual regime: mean revert when choppy, trend follow when trending
8. RELAXED thresholds: CRSI<20/>80 (not 10/90), RSI<35/>65 (not 30/70)
9. FALLBACK entries: extreme RSI(2)<5/>95 alone guarantees trades

Why Connors RSI:
- CRSI = (RSI(2) + RSI_Streak(2) + PercentRank(100)) / 3
- Long when CRSI < 20 (oversold), Short when CRSI > 80 (overbought)
- 75% win rate reported through 2022 crash
- Works in bear markets where trend strategies fail

Key changes from failed 12h strategies:
- CRSI instead of simple RSI (better mean reversion signal)
- RELAXED thresholds to ensure trades (CRSI 20/80 not 10/90)
- Fallback: RSI(2) extremes alone trigger entries
- Hold logic maintains position through minor pullbacks
- 1d HTF for bias only — not entry filter (reduces 0-trade risk)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 25-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_hma_1d_atr_fallback_v1"
timeframe = "12h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_connors_rsi(close, rsi_period=2, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(2) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(2): Very short-term momentum
    RSI_Streak(2): RSI of consecutive up/down days
    PercentRank(100): Where current close ranks vs last 100 closes
    
    CRSI < 20 = oversold (long), CRSI > 80 = overbought (short)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(2)
    rsi_2 = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 else -1
        else:
            streak[i] = streak[i-1] if i > 0 else 0
    
    # Convert streak to absolute for RSI calculation
    streak_for_rsi = np.abs(streak)
    streak_rsi = calculate_rsi(streak_for_rsi, streak_period)
    
    # Percent Rank
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_2[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_2[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], np.abs(high[j] - prev_close), np.abs(low[j] - prev_close))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    rsi_2 = calculate_rsi(close, period=2)  # For Connors and fallback
    crsi_12h = calculate_connors_rsi(close, rsi_period=2, streak_period=2, rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 1d HMA for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(crsi_12h[i]) or np.isnan(chop_12h[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if np.isnan(rsi_2[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === SECULAR TREND FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        ranging_regime = chop_12h[i] > 55
        trending_regime = chop_12h[i] < 45
        neutral_regime = not ranging_regime and not trending_regime
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_12h[i] < 35
        rsi_overbought = rsi_12h[i] > 65
        rsi_extreme_oversold = rsi_12h[i] < 25
        rsi_extreme_overbought = rsi_12h[i] > 75
        
        # RSI(2) extremes for fallback entries
        rsi2_extreme_oversold = rsi_2[i] < 5
        rsi2_extreme_overbought = rsi_2[i] > 95
        rsi2_oversold = rsi_2[i] < 10
        rsi2_overbought = rsi_2[i] > 90
        
        # === CONNORS RSI SIGNALS (Relaxed thresholds for trade count) ===
        crsi_oversold = crsi_12h[i] < 20
        crsi_overbought = crsi_12h[i] > 80
        crsi_extreme_oversold = crsi_12h[i] < 15
        crsi_extreme_overbought = crsi_12h[i] > 85
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Primary: CRSI oversold/overbought with ANY trend alignment
            if crsi_oversold and (above_sma200 or trend_1d_bullish):
                desired_signal = BASE_SIZE
            elif crsi_overbought and (below_sma200 or trend_1d_bearish):
                desired_signal = -BASE_SIZE
            
            # Fallback 1: CRSI extreme alone (guarantees trades)
            elif crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            elif crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            
            # Fallback 2: RSI(2) extreme alone (guarantees trades on all symbols)
            elif rsi2_extreme_oversold:
                desired_signal = REDUCED_SIZE
            elif rsi2_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            
            # Fallback 3: RSI(14) extreme with SMA200 filter
            elif rsi_extreme_oversold and above_sma200:
                desired_signal = REDUCED_SIZE
            elif rsi_extreme_overbought and below_sma200:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + CRSI pullback or RSI(2) oversold
            if trend_1d_bullish or above_sma200:
                if crsi_oversold or rsi2_oversold:
                    desired_signal = BASE_SIZE
                elif rsi_oversold:
                    desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + CRSI pullback or RSI(2) overbought
            if trend_1d_bearish or below_sma200:
                if crsi_overbought or rsi2_overbought:
                    desired_signal = -BASE_SIZE
                elif rsi_overbought:
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: CRSI + RSI confluence
            if crsi_oversold and rsi_oversold:
                desired_signal = REDUCED_SIZE
            elif crsi_overbought and rsi_overbought:
                desired_signal = -REDUCED_SIZE
            
            # Fallback: RSI(2) extremes
            elif rsi2_extreme_oversold:
                desired_signal = REDUCED_SIZE
            elif rsi2_extreme_overbought:
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and CRSI not overbought
                if (trend_1d_bullish or above_sma200) and crsi_12h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if (trend_1d_bearish or below_sma200) and crsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses + CRSI overbought
            if trend_1d_bearish and below_sma200 and crsi_12h[i] > 80:
                desired_signal = 0.0
            # Exit if RSI extremely overbought in ranging regime
            if ranging_regime and rsi_12h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses + CRSI oversold
            if trend_1d_bullish and above_sma200 and crsi_12h[i] < 20:
                desired_signal = 0.0
            # Exit if RSI extremely oversold in ranging regime
            if ranging_regime and rsi_12h[i] < 20:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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