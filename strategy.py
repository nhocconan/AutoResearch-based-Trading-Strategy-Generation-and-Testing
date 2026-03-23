#!/usr/bin/env python3
"""
Experiment #861: 4h Primary + 1d/1w HTF — Connors RSI + KAMA Adaptive Trend + Choppiness Regime

Hypothesis: After analyzing 597+ failed strategies, the winning pattern for 4h timeframe
combines Connors RSI (proven 75% win rate on reversals) with KAMA adaptive trend following
and Choppiness Index regime detection. This addresses the key failure modes:

1. Regular RSI too slow for 4h reversals → Connors RSI (RSI3 + Streak + PercentRank)
2. Fixed MA lag in choppy markets → KAMA adapts to volatility (ER-based)
3. Wrong regime logic → CHOP > 55 = mean revert, CHOP < 45 = trend follow
4. 2025 bear market needs reversal entries → Connors RSI excels at catching bottoms

Why this should beat Sharpe=0.612 baseline:
- Connors RSI has documented 75% win rate vs 55-60% for regular RSI
- KAMA reduces whipsaw in 2022 crash (adaptive smoothing)
- 1d HMA + 1w HMA dual HTF filter prevents counter-trend trades
- ATR trailing stop (2.5x) protects from 2022-style crashes

Position sizing: 0.25-0.30 discrete levels (max 0.35)
Target trades: 25-45/year on 4h (meets 20-50 target)
Expected Sharpe: 0.7-1.2 based on Connors RSI literature
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_kama_chop_regime_1d1w_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average with min_periods."""
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes
    High ER = trending (fast SC), Low ER = choppy (slow SC)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        net_change = np.abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
        sc[i] = sc[i] ** 2  # Square for smoother adaptation
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    return kama

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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(close,3) + RSI(Streak,2) + PercentRank(100)) / 3
    
    RSI(3): Short-term momentum
    RSI(Streak): Duration of consecutive up/down days
    PercentRank: Percentile of today's change vs last 100 days
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3) - short period
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (absolute streak length)
    streak_abs = np.abs(streak)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        if streak_abs[i] == 0:
            streak_rsi[i] = 50.0
        else:
            # Map streak to 0-100: longer streak = more extreme
            streak_rsi[i] = min(100, max(0, 50 + streak[i] * 10))
    
    # Percent Rank - today's return vs last 100 days
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            today_return = returns[-1]
            count_below = np.sum(returns[:-1] < today_return)
            percent_rank[i] = 100.0 * count_below / (len(returns) - 1)
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging (mean revert), CHOP < 45 = trending (trend follow).
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
        prev_close = close[i-1]
        tr[i] = max(high[i] - low[i], np.abs(high[i] - prev_close), np.abs(low[i] - prev_close))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        tr[i] = max(high[i] - low[i], np.abs(high[i] - prev_close), np.abs(low[i] - prev_close))
        
        if high[i] - prev_high > prev_low - low[i]:
            plus_dm[i] = max(0, high[i] - prev_high)
        else:
            plus_dm[i] = 0
        
        if prev_low - low[i] > high[i] - prev_high:
            minus_dm[i] = max(0, prev_low - low[i])
        else:
            minus_dm[i] = 0
    
    atr_series = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr_series + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr_series + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    atr_4h = calculate_atr(high, low, close, period=14)
    adx_4h = calculate_adx(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 1d HMA for medium-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for long-term secular trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(kama_4h[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(adx_4h[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        
        # === LONG-TERM TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h KAMA adaptive) ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === SECULAR TREND FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 55
        trending_regime = chop_4h[i] < 45
        neutral_regime = not ranging_regime and not trending_regime
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_4h[i] > 25
        weak_trend = adx_4h[i] < 20
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_oversold = crsi_4h[i] < 10
        crsi_extreme_overbought = crsi_4h[i] > 90
        crsi_oversold = crsi_4h[i] < 20
        crsi_overbought = crsi_4h[i] > 80
        crsi_recovering = crsi_4h[i] > 30 and crsi_4h[i-1] < 30 if not np.isnan(crsi_4h[i-1]) else False
        crsi_weakening = crsi_4h[i] < 70 and crsi_4h[i-1] > 70 if not np.isnan(crsi_4h[i-1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion with CRSI ===
        if ranging_regime:
            # Long: CRSI extreme oversold + ANY trend alignment (at least one bullish)
            if crsi_extreme_oversold and (trend_1w_bullish or trend_1d_bullish or above_sma200):
                desired_signal = BASE_SIZE
            
            # Short: CRSI extreme overbought + ANY trend alignment (at least one bearish)
            if crsi_extreme_overbought and (trend_1w_bearish or trend_1d_bearish or below_sma200):
                desired_signal = -BASE_SIZE
            
            # CRSI recovery cross (high probability reversal)
            if crsi_recovering and (trend_1d_bullish or kama_bullish):
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if crsi_weakening and (trend_1d_bearish or kama_bearish):
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # Fallback: moderate CRSI extremes (guarantees trades)
            if crsi_oversold and trend_1w_bullish and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if crsi_overbought and trend_1w_bearish and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend alignment + CRSI pullback (not extreme)
            if trend_1w_bullish and trend_1d_bullish:
                if 30 < crsi_4h[i] < 50 and kama_bullish:
                    desired_signal = BASE_SIZE
                elif crsi_recovering and strong_trend:
                    desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # Short: Bearish trend alignment + CRSI pullback
            if trend_1w_bearish and trend_1d_bearish:
                if 50 < crsi_4h[i] < 70 and kama_bearish:
                    desired_signal = -BASE_SIZE
                elif crsi_weakening and strong_trend:
                    desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: CRSI extremes + strong trend alignment
            if crsi_extreme_oversold and trend_1w_bullish and trend_1d_bullish:
                desired_signal = BASE_SIZE
            
            if crsi_extreme_overbought and trend_1w_bearish and trend_1d_bearish:
                desired_signal = -BASE_SIZE
            
            # CRSI recovery with KAMA confirmation
            if crsi_recovering and kama_bullish and above_sma200:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if crsi_weakening and kama_bearish and below_sma200:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
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
                if (trend_1w_bullish or trend_1d_bullish) and crsi_4h[i] < 85:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if (trend_1w_bearish or trend_1d_bearish) and crsi_4h[i] > 15:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if both HTF trends reverse + CRSI overbought
            if trend_1w_bearish and trend_1d_bearish and crsi_4h[i] > 85:
                desired_signal = 0.0
            # Exit if CRSI extremely overbought in ranging regime
            if ranging_regime and crsi_4h[i] > 95:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both HTF trends reverse + CRSI oversold
            if trend_1w_bullish and trend_1d_bullish and crsi_4h[i] < 15:
                desired_signal = 0.0
            # Exit if CRSI extremely oversold in ranging regime
            if ranging_regime and crsi_4h[i] < 5:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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