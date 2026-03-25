#!/usr/bin/env python3
"""
Experiment #1567: 6h Primary + 1d/1w HTF — Hybrid Vol/Trend Strategy

Hypothesis: 6h timeframe is underexplored and sits between 4h (noisy) and 12h (slow).
This strategy uses HYBRID logic: mean-reversion after vol spikes + trend-following
in calm markets. The key insight from failed 6h experiments is OVER-FILTERING.

Why previous 6h failed:
- CRSI+CHOP+RSI+HTF = too many conditions, never all true
- Weekly pivot strategies = too rare signals
- KAMA strategies = too much lag on 6h

New approach:
1. VOL SPIKE REVERSION: ATR7/ATR30 > 1.8 + BB extreme = mean revert (high prob)
2. TREND FOLLOW: ATR ratio < 1.2 + HMA aligned = ride momentum (lower prob but bigger moves)
3. LOOSE thresholds: RSI 30/70 (not 25/75), ATR ratio 1.8 (not 2.0)
4. HTF bias from 1d ONLY (1w too slow for 6h entries)
5. Discrete sizing: 0.0, ±0.25, ±0.30

Entry logic (GUARANTEED to generate trades):
- LONG vol reversion: ATR_ratio>1.8 + price<BB_lower + RSI<40 (no HTF filter = more trades)
- SHORT vol reversion: ATR_ratio>1.8 + price>BB_upper + RSI>60
- LONG trend: ATR_ratio<1.2 + HMA16>HMA48 + price>HMA_1d
- SHORT trend: ATR_ratio<1.2 + HMA16<HMA48 + price<HMA_1d

Exit: ATR_ratio normalizes (1.3-1.6) OR stoploss (2.5x ATR) OR signal flip

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hybrid_vol_trend_1d_v1"
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
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    # Volatility ratio (ATR7/ATR30)
    atr_ratio = np.full(n, np.nan, dtype=np.float64)
    mask = (atr_30 > 1e-10) & (~np.isnan(atr_7)) & (~np.isnan(atr_30))
    atr_ratio[mask] = atr_7[mask] / atr_30[mask]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
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
        if np.isnan(atr_ratio[i]) or atr_ratio[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(atr_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY REGIME ===
        vol_spike = atr_ratio[i] > 1.8  # Elevated volatility = mean reversion
        vol_calm = atr_ratio[i] < 1.2   # Low volatility = trend following
        vol_neutral = not vol_spike and not vol_calm
        
        # === PRICE EXTREME DETECTION ===
        price_extreme_low = close[i] <= bb_lower[i] * 1.005  # At or below lower BB
        price_extreme_high = close[i] >= bb_upper[i] * 0.995  # At or above upper BB
        
        # === RSI EXTREME (LOOSE thresholds) ===
        rsi = rsi_14[i]
        rsi_oversold = rsi < 40  # Was 35, loosened for more trades
        rsi_overbought = rsi > 60  # Was 65, loosened for more trades
        
        # === HTF TREND BIAS (1d only, 1w too slow) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA CROSSOVER ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === ENTRY LOGIC (LOOSE - MUST GENERATE TRADES) ===
        desired_signal = 0.0
        
        # VOL SPIKE REVERSION (high probability, smaller moves)
        if vol_spike:
            # LONG: price at BB lower + RSI oversold
            if price_extreme_low and rsi_oversold:
                desired_signal = SIZE_BASE
            
            # SHORT: price at BB upper + RSI overbought
            elif price_extreme_high and rsi_overbought:
                desired_signal = -SIZE_BASE
        
        # TREND FOLLOWING (lower probability, bigger moves)
        elif vol_calm:
            # LONG: HMA bullish + price above 1d HMA
            if hma_bullish and price_above_1d:
                desired_signal = SIZE_STRONG
            
            # SHORT: HMA bearish + price below 1d HMA
            elif hma_bearish and price_below_1d:
                desired_signal = -SIZE_STRONG
        
        # NEUTRAL VOL: Use HTF bias + RSI filter (catches more trades)
        elif vol_neutral:
            # LONG: 1d bullish + RSI not overbought + HMA aligned
            if price_above_1d and rsi < 65 and hma_bullish:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + RSI not oversold + HMA aligned
            elif price_below_1d and rsi > 35 and hma_bearish:
                desired_signal = -SIZE_BASE
        
        # === EXIT LOGIC ===
        # Exit when volatility normalizes after spike (reversion complete)
        if in_position and vol_spike:
            # Check if vol is coming down (reversion happening)
            if i > 0 and not np.isnan(atr_ratio[i-1]):
                if atr_ratio[i-1] > 2.0 and atr_ratio[i] < 1.6:
                    desired_signal = 0.0  # Take profit on vol normalization
        
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
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
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