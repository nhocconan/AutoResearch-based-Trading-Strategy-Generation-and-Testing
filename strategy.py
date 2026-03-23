#!/usr/bin/env python3
"""
Experiment #879: 4h Primary + 1d HTF — KAMA Adaptive Trend + Choppiness Regime + RSI

Hypothesis: After 600+ failed strategies, KAMA (Kaufman Adaptive Moving Average) provides
superior trend adaptation compared to static EMAs/SMAs. KAMA adjusts smoothing based on
market efficiency ratio (ER), reducing whipsaws in choppy markets while capturing trends.

Key insights from research:
1. KAMA + ADX + Choppiness filter achieved ETH Sharpe +0.755 in backtests
2. 4h timeframe targets 20-50 trades/year (optimal fee/trade balance)
3. 1d HMA(21) provides strong HTF trend bias without overfitting
4. Choppiness Index(14) cleanly separates regime: CHOP>55=range, CHOP<45=trend
5. RSI(14) extremes provide entry timing within regime-appropriate logic
6. ATR(14) 2.5x trailing stop protects against adverse moves

Why KAMA over HMA/EMA:
- KAMA adapts smoothing constant based on price directionality (Efficiency Ratio)
- ER = |net_change| / sum(|individual_changes|) over period
- ER near 1 = strong trend (fast KAMA), ER near 0 = noise (slow KAMA)
- This naturally reduces whipsaws in 2022 crash and 2025 bear market

Strategy Logic:
- RANGING (CHOP>55): Mean reversion via RSI extremes (long RSI<30, short RSI>70)
- TRENDING (CHOP<45): Trend follow via KAMA slope + 1d HMA bias
- NEUTRAL (45-55): Conservative, require both RSI extreme + HTF alignment

Position Sizing:
- BASE_SIZE = 0.28 (28% of capital)
- REDUCED_SIZE = 0.20 (20% for lower conviction)
- Discrete levels minimize fee churn from signal changes
- Stoploss: 2.5x ATR trailing from entry/extreme

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_chop_regime_rsi_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    
    Adapts smoothing based on market Efficiency Ratio (ER).
    ER = |price_change_over_period| / sum(|individual_price_changes|)
    
    When ER is high (trending), KAMA follows price closely.
    When ER is low (choppy), KAMA flattens to reduce whipsaws.
    
    Constants:
    - fast_sc = 2/(fast_period+1) = 2/3 for fast smoothing
    - slow_sc = 2/(slow_period+1) = 2/31 for slow smoothing
    - sc = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + 1:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(period, n):
        net_change = np.abs(close[i] - close[i-period])
        sum_changes = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0
    
    # Calculate smoothing constant (sc)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]  # Initialize
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    rsi_4h = calculate_rsi(close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 1d HMA for HTF trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
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
        if np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === HTF TREND BIAS (1d HMA21) ===
        htm_bullish = close[i] > hma_1d_aligned[i]
        htm_bearish = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND DIRECTION (4h) ===
        kama_slope_bullish = kama_4h[i] > kama_4h[i-5] if not np.isnan(kama_4h[i-5]) else False
        kama_slope_bearish = kama_4h[i] < kama_4h[i-5] if not np.isnan(kama_4h[i-5]) else False
        price_above_kama = close[i] > kama_4h[i]
        price_below_kama = close[i] < kama_4h[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 55
        trending_regime = chop_4h[i] < 45
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        rsi_extreme_oversold = rsi_4h[i] < 25
        rsi_extreme_overbought = rsi_4h[i] > 75
        rsi_neutral = 35 <= rsi_4h[i] <= 65
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: RSI oversold + price below KAMA (pullback in range)
            if rsi_oversold and price_below_kama:
                desired_signal = BASE_SIZE
            # Short: RSI overbought + price above KAMA (rally in range)
            elif rsi_overbought and price_above_kama:
                desired_signal = -BASE_SIZE
            # Fallback: Extreme RSI alone (ensures trade generation)
            elif rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            elif rsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: HTF bullish + KAMA slope up + price above KAMA
            if htm_bullish and kama_slope_bullish and price_above_kama:
                # Enter on RSI pullback (not extreme)
                if rsi_4h[i] < 55:
                    desired_signal = BASE_SIZE
                elif rsi_neutral:
                    desired_signal = REDUCED_SIZE
            # Short: HTF bearish + KAMA slope down + price below KAMA
            elif htm_bearish and kama_slope_bearish and price_below_kama:
                # Enter on RSI bounce (not extreme)
                if rsi_4h[i] > 45:
                    desired_signal = -BASE_SIZE
                elif rsi_neutral:
                    desired_signal = -REDUCED_SIZE
            # Fallback: Strong HTF + KAMA alignment
            elif htm_bullish and price_above_kama and kama_slope_bullish:
                desired_signal = REDUCED_SIZE
            elif htm_bearish and price_below_kama and kama_slope_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) — Conservative ===
        else:
            # Require HTF alignment + RSI extreme
            if htm_bullish and rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            elif htm_bearish and rsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            # Secondary: KAMA + RSI confluence
            elif price_above_kama and rsi_oversold and kama_slope_bullish:
                desired_signal = REDUCED_SIZE
            elif price_below_kama and rsi_overbought and kama_slope_bearish:
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF bullish or KAMA support holds
                if htm_bullish and price_above_kama:
                    desired_signal = BASE_SIZE
                elif htm_bullish and rsi_4h[i] < 70:
                    desired_signal = REDUCED_SIZE
            elif position_side < 0:
                # Hold short if HTF bearish or KAMA resistance holds
                if htm_bearish and price_below_kama:
                    desired_signal = -BASE_SIZE
                elif htm_bearish and rsi_4h[i] > 30:
                    desired_signal = -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HTF reverses + RSI overbought
            if htm_bearish and rsi_4h[i] > 70:
                desired_signal = 0.0
            # Exit if extreme overbought in ranging regime
            if ranging_regime and rsi_extreme_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HTF reverses + RSI oversold
            if htm_bullish and rsi_4h[i] < 30:
                desired_signal = 0.0
            # Exit if extreme oversold in ranging regime
            if ranging_regime and rsi_extreme_oversold:
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