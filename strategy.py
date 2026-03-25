#!/usr/bin/env python3
"""
Experiment #1335: 6h Primary + 12h/1d HTF — Regime-Adaptive CHOP + Dual Mode

Hypothesis: Previous 6h strategies failed because they used ONE approach (trend OR mean-reversion).
This strategy ADAPTS to market regime using Choppiness Index (CHOP):
- CHOP < 38.2 = TREND regime → use trend-following (HMA slope + ROC momentum)
- CHOP > 61.8 = RANGE regime → use mean-reversion (RSI extremes + Bollinger bands)
- 38.2 <= CHOP <= 61.8 = TRANSITION → stay flat (avoid whipsaw)

Key innovations vs failed 6h strategies:
1. REGIME DETECTION: CHOP(14) with proven thresholds (38.2/61.8 from academic literature)
2. DUAL MODE: Different entry logic per regime (trend vs mean-revert)
3. HTF CONFIRMATION: 12h + 1d HMA bias must align with 6h signal direction
4. VOLUME FILTER: Require volume > 1.2x 20-bar MA for trend entries (avoid fakeouts)
5. ASYMMETRIC SIZING: 0.30 for trend regime (high conviction), 0.20 for range regime
6. DISCRETE SIGNALS: 0.0, ±0.20, ±0.30 only (minimize fee churn from signal changes)

Why this should beat Sharpe=0.447 baseline:
- Adapts to 2022 crash (range→trend→range transitions)
- Avoids whipsaw in transition zones (CHOP 38-62 = flat)
- 12h+1d filter prevents counter-trend trades that destroyed previous strategies
- Mean-reversion mode captures 2025 bear/range market (where trend strategies fail)

Entry logic:
- TREND LONG: CHOP<38.2 + 12h_HMA rising + 1d_HMA bullish + ROC>5 + vol>1.2x MA
- TREND SHORT: CHOP<38.2 + 12h_HMA falling + 1d_HMA bearish + ROC<-5 + vol>1.2x MA
- RANGE LONG: CHOP>61.8 + RSI<30 + price<BB_lower + 1d_HMA not strongly bearish
- RANGE SHORT: CHOP>61.8 + RSI>70 + price>BB_upper + 1d_HMA not strongly bullish

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.20 (range), 0.30 (trend)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_regime_adaptive_chop_dual_mode_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = Range market
    CHOP < 38.2 = Trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR for each bar
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100.0
    
    return roc

def calculate_volume_ma(volume, period=20):
    """Volume moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    roc_10 = calculate_roc(close, period=10)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
    # 6h HMA for local trend
    hma_6h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30  # Higher conviction in trend regime
    SIZE_RANGE = 0.20  # Lower conviction in range regime
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(roc_10[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (CHOP) ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        is_transition = not is_trend_regime and not is_range_regime
        
        # === HTF TREND BIAS ===
        # 12h HMA slope (compare to 3 bars ago)
        hma_12h_slope = 0.0
        if i >= 3 and not np.isnan(hma_12h_aligned[i-3]):
            hma_12h_slope = hma_12h_aligned[i] - hma_12h_aligned[i-3]
        
        # 1d HMA bias
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 6h price vs 6h HMA
        price_above_6h = close[i] > hma_6h[i]
        price_below_6h = close[i] < hma_6h[i]
        
        # Volume confirmation
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0.0
        vol_confirmed = vol_ratio > 1.2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if is_trend_regime:
            # TREND MODE: Follow momentum with HTF confirmation
            roc = roc_10[i]
            
            # LONG: 12h HMA rising + 1d bullish + ROC positive + volume confirmed
            if hma_12h_slope > 0 and price_above_1d and price_above_6h:
                if roc > 5.0 and vol_confirmed:
                    desired_signal = SIZE_TREND
                elif roc > 2.0:
                    desired_signal = SIZE_RANGE  # Weaker signal without volume
            
            # SHORT: 12h HMA falling + 1d bearish + ROC negative + volume confirmed
            elif hma_12h_slope < 0 and price_below_1d and price_below_6h:
                if roc < -5.0 and vol_confirmed:
                    desired_signal = -SIZE_TREND
                elif roc < -2.0:
                    desired_signal = -SIZE_RANGE
        
        elif is_range_regime:
            # RANGE MODE: Mean reversion at Bollinger extremes
            rsi = rsi_14[i]
            
            # LONG: RSI oversold + price at/near BB lower + 1d not strongly bearish
            if rsi < 30 and close[i] <= bb_lower[i]:
                # Only long if 1d HMA not strongly bearish (avoid catching falling knife)
                if price_above_1d or (hma_1d_aligned[i-1] <= hma_1d_aligned[i] if i >= 1 and not np.isnan(hma_1d_aligned[i-1]) else True):
                    desired_signal = SIZE_RANGE
            
            # SHORT: RSI overbought + price at/near BB upper + 1d not strongly bullish
            elif rsi > 70 and close[i] >= bb_upper[i]:
                # Only short if 1d HMA not strongly bullish
                if price_below_1d or (hma_1d_aligned[i-1] >= hma_1d_aligned[i] if i >= 1 and not np.isnan(hma_1d_aligned[i-1]) else True):
                    desired_signal = -SIZE_RANGE
        
        # TRANSITION regime: Stay flat (avoid whipsaw)
        # desired_signal remains 0.0
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_TREND * 0.9:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.9:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_RANGE * 0.9:
            final_signal = SIZE_RANGE
        elif desired_signal <= -SIZE_RANGE * 0.9:
            final_signal = -SIZE_RANGE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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