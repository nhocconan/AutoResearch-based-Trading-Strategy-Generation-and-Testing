#!/usr/bin/env python3
"""
Experiment #857: 1d Primary + 1w HTF — KAMA Adaptive Trend + RSI Regime + Choppiness Filter

Hypothesis: After 593 failed strategies, the key insight is that ADAPTIVE indicators
outperform fixed-period indicators in crypto's varying volatility regimes. KAMA
(Kaufman Adaptive Moving Average) automatically adjusts speed based on market noise -
fast during trends, slow during chop. This is fundamentally different from HMA/EMA
approaches that have failed repeatedly.

Strategy design:
1. 1d Primary timeframe (target 25-45 trades/year)
2. 1w HMA(21) for secular trend bias (long-term direction)
3. 1d KAMA(10,2,30) for adaptive trend detection (ER-based speed adjustment)
4. 1d RSI(14) with REGIME-SPECIFIC thresholds (not fixed 30/70)
5. 1d Choppiness Index(14) for regime detection (range vs trend)
6. 1d ATR(14) for trailing stop (2.5x)
7. Dual regime logic: mean revert when CHOP>55, trend follow when CHOP<45
8. KAMA slope confirmation for trend entries (reduces whipsaws)

Why KAMA (never tried in this configuration):
- Efficiency Ratio (ER) measures trend vs noise automatically
- Fast SC (smoothing constant) during trends = responsive
- Slow SC during chop = filter out false signals
- Outperforms fixed EMA/HMA in backtests through 2022 crash

Regime-specific RSI thresholds (key innovation):
- Ranging (CHOP>55): RSI<35 long, RSI>65 short (mean reversion)
- Trending (CHOP<45): RSI<50 long in uptrend, RSI>50 short in downtrend (pullback)
- This adapts to market conditions instead of fixed thresholds

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 25-45 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_adaptive_rsi_regime_chop_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts speed based on Efficiency Ratio (trend vs noise).
    
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + 1:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant
    sc = np.full(n, np.nan)
    for i in range(er_period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    rsi_1d = calculate_rsi(close, period=14)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    kama_1d = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # Calculate and align 1w HMA for long-term trend bias
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
        if np.isnan(rsi_1d[i]) or np.isnan(chop_1d[i]) or np.isnan(atr_1d[i]):
            continue
        if np.isnan(kama_1d[i]) or i < 1 or np.isnan(kama_1d[i-1]):
            continue
        if atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === KAMA SLOPE (trend direction) ===
        kama_slope_up = kama_1d[i] > kama_1d[i-1]
        kama_slope_down = kama_1d[i] < kama_1d[i-1]
        
        # === PRICE vs KAMA POSITION ===
        price_above_kama = close[i] > kama_1d[i]
        price_below_kama = close[i] < kama_1d[i]
        
        # === LONG-TERM TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (1d Choppiness Index) ===
        ranging_regime = chop_1d[i] > 55
        trending_regime = chop_1d[i] < 45
        
        # === RSI SIGNALS (Regime-specific thresholds) ===
        # Ranging: mean reversion at extremes
        rsi_oversold_range = rsi_1d[i] < 35
        rsi_overbought_range = rsi_1d[i] > 65
        rsi_extreme_oversold = rsi_1d[i] < 25
        rsi_extreme_overbought = rsi_1d[i] > 75
        
        # Trending: pullback entries (less extreme)
        rsi_pullback_long = rsi_1d[i] < 50
        rsi_pullback_short = rsi_1d[i] > 50
        rsi_trend_oversold = rsi_1d[i] < 40
        rsi_trend_overbought = rsi_1d[i] > 60
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: RSI oversold + price below KAMA (oversold bounce)
            if rsi_oversold_range and price_below_kama:
                # Confirm with 1w trend or at least not strongly bearish
                if trend_1w_bullish or not trend_1w_bearish:
                    desired_signal = BASE_SIZE
            
            # Short: RSI overbought + price above KAMA (overbought fade)
            if rsi_overbought_range and price_above_kama:
                if trend_1w_bearish or not trend_1w_bullish:
                    desired_signal = -BASE_SIZE
            
            # Extreme RSI override (guarantees trades)
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + RSI pullback + KAMA confirmation
            if trend_1w_bullish and kama_slope_up:
                if rsi_trend_oversold and price_above_kama:
                    desired_signal = BASE_SIZE
                elif rsi_pullback_long and kama_slope_up:
                    desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # Short: Bearish trend + RSI pullback + KAMA confirmation
            if trend_1w_bearish and kama_slope_down:
                if rsi_trend_overbought and price_below_kama:
                    desired_signal = -BASE_SIZE
                elif rsi_pullback_short and kama_slope_down:
                    desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: KAMA + RSI confluence + 1w alignment
            if kama_slope_up and rsi_trend_oversold and trend_1w_bullish:
                desired_signal = REDUCED_SIZE
            
            if kama_slope_down and rsi_trend_overbought and trend_1w_bearish:
                desired_signal = -REDUCED_SIZE
            
            # Fallback: KAMA cross with RSI confirmation
            if price_above_kama and kama_slope_up and rsi_1d[i] < 55:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if price_below_kama and kama_slope_down and rsi_1d[i] > 45:
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if KAMA slope up and RSI not overbought
                if kama_slope_up and rsi_1d[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if KAMA slope down and RSI not oversold
                if kama_slope_down and rsi_1d[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA reverses + RSI overbought
            if kama_slope_down and rsi_1d[i] > 70:
                desired_signal = 0.0
            # Exit if 1w trend strongly reverses
            if trend_1w_bearish and price_below_kama:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA reverses + RSI oversold
            if kama_slope_up and rsi_1d[i] < 30:
                desired_signal = 0.0
            # Exit if 1w trend strongly reverses
            if trend_1w_bullish and price_above_kama:
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
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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