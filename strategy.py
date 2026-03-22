#!/usr/bin/env python3
"""
Experiment #107: 12h Regime-Adaptive Strategy with Choppiness Index + Multi-TF Filter

Hypothesis: Based on research showing BTC/ETH fail simple trend strategies in bear/range markets,
this strategy adapts to market regime using Choppiness Index (CHOP):
- CHOP > 61.8: Range regime → Mean reversion (RSI extremes + Bollinger Bands)
- CHOP < 38.2: Trend regime → Trend following (KAMA + 1d HMA bias)
- 38.2 <= CHOP <= 61.8: Transition → Reduced position size or flat

Key innovations:
1. Choppiness Index for regime detection (proven edge in bear markets)
2. 1d HMA for higher-timeframe trend bias (prevents counter-trend trades)
3. Asymmetric logic: More aggressive shorts in bear regime (2025 test is bearish)
4. ATR-based stoploss (2.5*ATR) to protect against adverse moves
5. Discrete position sizing (0.20/0.30) to minimize fee churn

Why 12h timeframe:
- Captures medium-term trends without 4h/1h noise
- Fewer trades = less fee drag (target 30-50 trades/year)
- Better suited for regime detection than lower timeframes

HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_1d_hma_kama_rsi_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - moves fast in trends, slow in ranges.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    price_change = np.abs(close - np.roll(close, er_period))
    price_change[:er_period] = np.nan
    
    individual_changes = np.abs(np.diff(close))
    individual_changes = np.insert(individual_changes, 0, 0)
    
    sum_changes = pd.Series(individual_changes).rolling(window=er_period, min_periods=er_period).sum().values
    
    er = np.zeros(n)
    mask = sum_changes > 0
    er[mask] = price_change[mask] / sum_changes[mask]
    er[:er_period] = np.nan
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc[:er_period] = np.nan
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = np.nan
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range-bound market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        # Calculate ATR for each bar in the lookback window
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            atr_sum += tr
        
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100  # No losses = RSI 100
    
    return rsi

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return sma, upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    chop = calculate_choppiness(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_sma, bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = range (mean reversion)
        # CHOP < 38.2 = trend (trend following)
        range_regime = chop[i] > 61.8
        trend_regime = chop[i] < 38.2
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # KAMA slope (trend strength)
        kama_slope = 0.0
        if i >= 5 and not np.isnan(kama[i-5]):
            kama_slope = (kama[i] - kama[i-5]) / kama[i-5] if kama[i-5] != 0 else 0
        
        new_signal = 0.0
        
        # === TREND REGIME (CHOP < 38.2) ===
        if trend_regime:
            # Long: 1d bullish + KAMA bullish + positive slope
            if bull_trend_1d and kama_bullish and kama_slope > 0:
                new_signal = SIZE_STRONG
            elif bull_trend_1d and kama_bullish:
                new_signal = SIZE_BASE
            elif kama_bullish and kama_slope > 0:
                new_signal = SIZE_BASE
            
            # Short: 1d bearish + KAMA bearish + negative slope
            if bear_trend_1d and kama_bearish and kama_slope < 0:
                new_signal = -SIZE_STRONG
            elif bear_trend_1d and kama_bearish:
                new_signal = -SIZE_BASE
            elif kama_bearish and kama_slope < 0:
                new_signal = -SIZE_BASE
        
        # === RANGE REGIME (CHOP > 61.8) - MEAN REVERSION ===
        elif range_regime:
            # Long: RSI oversold + price near lower BB + 1d not strongly bearish
            rsi_oversold = rsi[i] < 30
            price_near_lower_bb = close[i] <= bb_lower[i] * 1.01  # Within 1% of lower BB
            
            if rsi_oversold and price_near_lower_bb and not bear_trend_1d:
                new_signal = SIZE_BASE
            elif rsi_oversold and price_near_lower_bb:
                new_signal = SIZE_BASE * 0.5  # Reduced size against HTF trend
            
            # Short: RSI overbought + price near upper BB + 1d not strongly bullish
            rsi_overbought = rsi[i] > 70
            price_near_upper_bb = close[i] >= bb_upper[i] * 0.99  # Within 1% of upper BB
            
            if rsi_overbought and price_near_upper_bb and not bull_trend_1d:
                new_signal = -SIZE_BASE
            elif rsi_overbought and price_near_upper_bb:
                new_signal = -SIZE_BASE * 0.5  # Reduced size against HTF trend
        
        # === TRANSITION REGIME (38.2 <= CHOP <= 61.8) ===
        # Reduced position size or flat
        else:
            # Only take strong signals in transition
            if bull_trend_1d and kama_bullish and kama_slope > 0.02:
                new_signal = SIZE_BASE * 0.5
            elif bear_trend_1d and kama_bearish and kama_slope < -0.02:
                new_signal = -SIZE_BASE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals