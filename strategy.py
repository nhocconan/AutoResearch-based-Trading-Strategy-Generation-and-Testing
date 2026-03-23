#!/usr/bin/env python3
"""
Experiment #226: 12h Primary + 1d HTF — Volatility Regime + Mean Reversion + Trend Confluence

Hypothesis: After 12h failures with HMA+Donchian (#216) and KAMA+Choppiness+CRSI (#222),
try a VOLATILITY-ADJUSTED regime approach that adapts to market conditions:

Key innovations vs failed attempts:
1. CHOPPINESS INDEX (14) for regime: >61.8 = range (mean revert), <38.2 = trend (follow)
2. RSI(3) for fast mean reversion signals (Connors-style, not RSI14)
3. ATR RATIO (ATR7/ATR30) for vol spike detection — only enter when vol elevated
4. 1d HMA(21) macro bias filter (aligned via mtf_data)
5. ADX hysteresis (enter >25, exit <18) to reduce whipsaw
6. Asymmetric sizing: full size with macro trend, half against

Why this might work on 12h:
- Higher TF = fewer false signals, better regime detection
- Vol spike filter avoids entering during low-vol chop (fee drain)
- RSI(3) catches quick reversals that RSI(14) misses
- CHOP filter switches between mean-revert and trend-follow dynamically

TARGET: 25-40 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
Position sizing: 0.0, ±0.20, ±0.30 (discrete to minimize fee churn)
Stoploss: 2.5x ATR(14) trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_volregime_chop_rsi3_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    n = len(close)
    close_s = pd.Series(close)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_atr = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_atr = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * plus_atr[i] / atr[i]
            minus_di[i] = 100.0 * minus_atr[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    rsi_3 = calculate_rsi(close, period=3)  # Fast RSI for mean reversion
    rsi_14 = calculate_rsi(close, period=14)  # Standard RSI for filter
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # Calculate 1d HMA for macro trend (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(rsi_3[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            continue
        
        # === VOLATILITY REGIME (ATR Ratio) ===
        atr_ratio = atr_7[i] / atr_30[i]
        vol_elevated = atr_ratio > 1.3  # Vol spike detected
        vol_normal = atr_ratio <= 1.3
        
        # === CHOPPINESS REGIME ===
        choppy_regime = chop_14[i] > 61.8  # Range-bound
        trending_regime = chop_14[i] < 38.2  # Strong trend
        neutral_regime = 38.2 <= chop_14[i] <= 61.8
        
        # === HTF MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND DETECTION (12h HMA crossover) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === TREND STRENGTH (ADX with hysteresis) ===
        adx_strong = adx_14[i] > 25.0
        adx_weak = adx_14[i] < 18.0
        
        # === RSI(3) MEAN REVERSION SIGNALS ===
        rsi3_oversold = rsi_3[i] < 15.0  # Extreme oversold
        rsi3_overbought = rsi_3[i] > 85.0  # Extreme overbought
        rsi3_neutral_long = 30.0 <= rsi_3[i] <= 50.0  # Pullback in uptrend
        rsi3_neutral_short = 50.0 <= rsi_3[i] <= 70.0  # Pullback in downtrend
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRIES
        if choppy_regime:
            # Mean reversion in range: buy extreme oversold
            if rsi3_oversold and vol_elevated:
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_FULL  # With macro trend
                else:
                    new_signal = POSITION_SIZE_HALF  # Against macro
        elif trending_regime:
            # Trend following: buy pullback in uptrend
            if hma_bullish and rsi3_neutral_long:
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_FULL
                else:
                    new_signal = POSITION_SIZE_HALF
            # Breakout entry
            elif hma_bullish and adx_strong and rsi_14[i] < 65.0:
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_FULL
        else:
            # Neutral regime: use HMA crossover + RSI filter
            if hma_bullish and rsi3_neutral_long and vol_normal:
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRIES
        if choppy_regime:
            # Mean reversion in range: sell extreme overbought
            if rsi3_overbought and vol_elevated:
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_FULL  # With macro trend
                else:
                    new_signal = -POSITION_SIZE_HALF  # Against macro
        elif trending_regime:
            # Trend following: sell pullback in downtrend
            if hma_bearish and rsi3_neutral_short:
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_FULL
                else:
                    new_signal = -POSITION_SIZE_HALF
            # Breakout entry
            elif hma_bearish and adx_strong and rsi_14[i] > 35.0:
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_FULL
        else:
            # Neutral regime: use HMA crossover + RSI filter
            if hma_bearish and rsi3_neutral_short and vol_normal:
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if HMA still bullish and RSI not overbought
                if hma_bullish and rsi_3[i] < 90.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if HMA still bearish and RSI not oversold
                if hma_bearish and rsi_3[i] > 10.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if HMA crosses bearish (with hysteresis)
        if in_position and position_side > 0 and hma_bearish and adx_weak:
            new_signal = 0.0
        
        # Exit short if HMA crosses bullish (with hysteresis)
        if in_position and position_side < 0 and hma_bullish and adx_weak:
            new_signal = 0.0
        
        # Exit if macro trend reverses against position
        if in_position and position_side > 0 and price_below_hma_1d and choppy_regime:
            new_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d and choppy_regime:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals