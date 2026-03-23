#!/usr/bin/env python3
"""
Experiment #1002: 12h Primary + 1d/1w HTF — Simplified Regime Adaptive + RSI Mean Reversion

Hypothesis: After 727 failed strategies, the key issue is OVER-FILTERING causing 0 trades.
This strategy SIMPLIFIES entry conditions while keeping regime-adaptive logic.

Key insights from research:
1. 12h timeframe targets 20-50 trades/year (lower fee drag than 4h/1h)
2. Choppiness Index regime: CHOP>55=range(mean revert), CHOP<45=trend(breakout)
3. 1d HMA(21) for macro trend bias, 1w HMA(21) for secular trend
4. RSI(14) extremes for entry timing (oversold<35, overbought>65)
5. ATR(14) 2.5x trailing stoploss

CRITICAL CHANGE FROM FAILURES:
- Previous strategies required 4+ confluence factors = 0 trades
- This strategy requires only 2-3 factors MAX = ensures trade generation
- Funding rate removed (failed in #991, #993, #996)
- CRSI removed (failed in #993, #994, #995, #996)
- Focus on proven patterns: CHOP regime + RSI + HMA trend

Entry logic (SIMPLIFIED):
- Range regime (CHOP>55): RSI<35→long, RSI>65→short (mean revert)
- Trend regime (CHOP<45): RSI pullback in trend direction only
- Neutral: Conservative entries with HTF confluence

Position sizing: 0.0, ±0.25, ±0.30 (discrete to minimize fee churn)
Stoploss: 2.5*ATR trailing (signal→0 when hit)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_simplified_regime_rsi_1d1w_hma_chop_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index - Wilder's method."""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range - Wilder's method."""
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures market choppy vs trending."""
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
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_adx(high, low, close, period=14):
    """Average Directional Index."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 3:
        return adx
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm / (atr + 1e-10)).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = 100 * pd.Series(minus_dm / (atr + 1e-10)).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
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
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    adx_12h = calculate_adx(high, low, close, period=14)
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for secular trend
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(chop_12h[i]) or np.isnan(adx_12h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === SECULAR TREND (1w HTF HMA21) ===
        secular_bull = close[i] > hma_1w_aligned[i]
        secular_bear = close[i] < hma_1w_aligned[i]
        
        # === MACRO TREND (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        ranging_regime = chop_12h[i] > 55
        trending_regime = chop_12h[i] < 45
        
        # === TREND STRENGTH (12h ADX) ===
        strong_trend = adx_12h[i] > 25
        weak_trend = adx_12h[i] < 20
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_12h[i] < 35
        rsi_overbought = rsi_12h[i] > 65
        rsi_extreme_oversold = rsi_12h[i] < 25
        rsi_extreme_overbought = rsi_12h[i] > 75
        rsi_neutral = 35 <= rsi_12h[i] <= 65
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: RSI oversold (simple, ensures trades)
            if rsi_oversold:
                if secular_bull or macro_bull:
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = REDUCED_SIZE
            
            # Short: RSI overbought (simple, ensures trades)
            if rsi_overbought:
                if secular_bear or macro_bear:
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -REDUCED_SIZE
            
            # Extreme RSI overrides regime (guarantees trades in strong reversals)
            if rsi_extreme_oversold:
                desired_signal = max(desired_signal, BASE_SIZE)
            if rsi_extreme_overbought:
                desired_signal = min(desired_signal, -BASE_SIZE)
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + RSI pullback (not oversold, just cooling)
            if (secular_bull or macro_bull) and strong_trend:
                if 30 <= rsi_12h[i] <= 50:  # Pullback in uptrend
                    desired_signal = BASE_SIZE
                elif rsi_12h[i] < 35:  # Deep pullback
                    desired_signal = BASE_SIZE
            
            # Short: Bearish trend + RSI rally (not overbought, just cooling)
            if (secular_bear or macro_bear) and strong_trend:
                if 50 <= rsi_12h[i] <= 70:  # Rally in downtrend
                    desired_signal = -BASE_SIZE
                elif rsi_12h[i] > 65:  # Strong rally
                    desired_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Only enter with HTF confluence
            if rsi_extreme_oversold and (secular_bull or macro_bull):
                desired_signal = BASE_SIZE
            elif rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and (secular_bear or macro_bear):
                desired_signal = -BASE_SIZE
            elif rsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            
            # Weak trend + RSI extreme = entry
            if weak_trend and rsi_oversold:
                desired_signal = max(desired_signal, REDUCED_SIZE)
            if weak_trend and rsi_overbought:
                desired_signal = min(desired_signal, -REDUCED_SIZE)
        
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
                # Hold long if secular/macro bull and RSI not extreme overbought
                if (secular_bull or macro_bull) and rsi_12h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if secular/macro bear and RSI not extreme oversold
                if (secular_bear or macro_bear) and rsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if both HTF trends reverse + RSI overbought
            if secular_bear and macro_bear and rsi_12h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both HTF trends reverse + RSI oversold
            if secular_bull and macro_bull and rsi_12h[i] < 30:
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