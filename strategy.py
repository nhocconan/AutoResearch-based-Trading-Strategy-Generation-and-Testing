#!/usr/bin/env python3
"""
Experiment #842: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + ADX + Choppiness Regime

Hypothesis: After 578+ failed strategies, the key insight is that 12h timeframe needs
ADAPTIVE trend detection (KAMA) that adjusts to market volatility, combined with
clear regime separation (Choppiness) and relaxed entry conditions to ensure trades
on ALL symbols (BTC, ETH, SOL).

Why 12h:
- Higher timeframe = fewer false signals, less fee drag
- Target 30-50 trades/year (manageable cost impact)
- Works in both bull and bear markets with regime detection

Strategy design:
1. 12h Primary timeframe
2. KAMA(21) for adaptive trend (adjusts ER based on volatility)
3. ADX(14) > 20 for trend strength confirmation
4. Choppiness(14) for regime: >55 range, <45 trend
5. RSI(14) for entry timing with relaxed thresholds (30/70)
6. 1d HMA(21) for HTF bias (aligns with higher timeframe trend)
7. 1w HMA(21) for secular trend filter
8. ATR(14) trailing stop (2.5x)
9. FALLBACK entries: extreme RSI (<20/>80) alone guarantees trades

Key changes from failed strategies:
- KAMA instead of HMA/EMA (adapts to volatility, less whipsaw)
- ADX filter for trend strength (avoid trading in weak trends)
- Relaxed RSI thresholds (30/70 not 25/75) for more signals
- FALLBACK: extreme RSI alone triggers entry (ensures trades on all symbols)
- Dual HTF: 1d for medium-term, 1w for long-term bias

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive
Timeframe: 12h (target 35-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_chop_regime_1d1w_rsi_atr_v1"
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

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise via Efficiency Ratio (ER).
    ER = |price change| / sum of absolute price changes
    High ER = trending (fast smoothing), Low ER = choppy (slow smoothing)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(period, n):
        price_change = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 1e-10:
            er[i] = price_change / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i-1]
            continue
        
        # Adaptive smoothing constant
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA formula
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX).
    Measures trend strength (not direction).
    ADX > 25 = strong trend, ADX < 20 = weak/ranging.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DX
    with np.errstate(divide='ignore', invalid='ignore'):
        di_sum = plus_di + minus_di + 1e-10
        dx = 100 * np.abs(plus_di - minus_di) / di_sum
    
    # ADX = SMA of DX
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx

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
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    kama_12h = calculate_kama(close, period=21)
    rsi_12h = calculate_rsi(close, period=14)
    adx_12h = calculate_adx(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
        if np.isnan(kama_12h[i]) or np.isnan(rsi_12h[i]) or np.isnan(adx_12h[i]):
            continue
        if np.isnan(chop_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === SECULAR TREND FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ADAPTIVE TREND (KAMA21) ===
        kama_bullish = close[i] > kama_12h[i]
        kama_bearish = close[i] < kama_12h[i]
        
        # === TREND STRENGTH (ADX14) ===
        strong_trend = adx_12h[i] > 20
        weak_trend = adx_12h[i] <= 20
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        ranging_regime = chop_12h[i] > 55
        trending_regime = chop_12h[i] < 45
        neutral_regime = not ranging_regime and not trending_regime
        
        # === RSI SIGNALS (Relaxed for 12h timeframe) ===
        rsi_oversold = rsi_12h[i] < 30
        rsi_overbought = rsi_12h[i] > 70
        rsi_extreme_oversold = rsi_12h[i] < 20
        rsi_extreme_overbought = rsi_12h[i] > 80
        rsi_neutral_low = 30 <= rsi_12h[i] < 45
        rsi_neutral_high = 55 < rsi_12h[i] <= 70
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: RSI oversold + any bullish bias
            if rsi_oversold and (trend_1w_bullish or trend_1d_bullish or above_sma200):
                desired_signal = BASE_SIZE
            
            # Short: RSI overbought + any bearish bias
            if rsi_overbought and (trend_1w_bearish or trend_1d_bearish or below_sma200):
                desired_signal = -BASE_SIZE
            
            # FALLBACK: Extreme RSI alone (guarantees trades on all symbols)
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
            
            # KAMA reversion in range
            if rsi_oversold and kama_bullish:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if rsi_overbought and kama_bearish:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Strong trend + bullish alignment + RSI pullback
            if strong_trend and (trend_1w_bullish or trend_1d_bullish):
                if rsi_neutral_low and kama_bullish:
                    desired_signal = BASE_SIZE
                elif rsi_oversold:
                    desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # Short: Strong trend + bearish alignment + RSI pullback
            if strong_trend and (trend_1w_bearish or trend_1d_bearish):
                if rsi_neutral_high and kama_bearish:
                    desired_signal = -BASE_SIZE
                elif rsi_overbought:
                    desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # FALLBACK: KAMA crossover with ADX confirmation
            if kama_bullish and strong_trend and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if kama_bearish and strong_trend and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: RSI + KAMA + HTF alignment
            if rsi_oversold and kama_bullish and (trend_1d_bullish or above_sma200):
                desired_signal = REDUCED_SIZE
            
            if rsi_overbought and kama_bearish and (trend_1d_bearish or below_sma200):
                desired_signal = -REDUCED_SIZE
            
            # FALLBACK: Extreme RSI with any trend alignment
            if rsi_extreme_oversold and (trend_1w_bullish or trend_1d_bullish):
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if rsi_extreme_overbought and (trend_1w_bearish or trend_1d_bearish):
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
                # Hold long if KAMA bullish and RSI not overbought
                if kama_bullish and rsi_12h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if KAMA bearish and RSI not oversold
                if kama_bearish and rsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA reverses + RSI overbought
            if kama_bearish and rsi_12h[i] > 75:
                desired_signal = 0.0
            # Exit if extreme overbought in ranging regime
            if ranging_regime and rsi_12h[i] > 85:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA reverses + RSI oversold
            if kama_bullish and rsi_12h[i] < 25:
                desired_signal = 0.0
            # Exit if extreme oversold in ranging regime
            if ranging_regime and rsi_12h[i] < 15:
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