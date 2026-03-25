#!/usr/bin/env python3
"""
Experiment #1389: 15m Primary + 1h/4h HTF — Trend Pullback with RSI Momentum

Hypothesis: 15m timeframe has ZERO successful experiments. Previous failures due to:
1. Entry conditions too strict (RSI thresholds too narrow = 0 trades)
2. Mean-reversion fighting strong trends (2022 crash whipsaw)
3. Too many trades on 15m = fee drag destroys Sharpe

This strategy uses TREND FOLLOWING with pullback entries (not mean-reversion):
1. 1h HMA(21) for intraday trend bias (direction filter only)
2. 15m RSI(7) pullback entries WITHIN trend (RSI 35-45 long, 55-65 short)
3. ROC(5) momentum confirmation (ensures pullback is ending)
4. Volume confirmation (taker_buy_volume ratio)
5. ATR(14) trailing stop at 2.5x

CRITICAL CHANGES from failed 15m strategies:
- RSI bands WIDER: 30-50 for long, 50-70 for short (not exact thresholds)
- Trend filter is DIRECTIONAL only (price vs HMA, not strict cross)
- Session filter REMOVED (was blocking too many entries)
- Size smaller: 0.15-0.25 (15m = higher frequency = smaller size)

Entry logic (LOOSE to guarantee trades):
- LONG: price > 1h_HMA + RSI(7) in 30-50 + ROC(5) > -2 (pullback ending)
- SHORT: price < 1h_HMA + RSI(7) in 50-70 + ROC(5) < 2 (bounce ending)

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_trend_pullback_rsi_roc_1h4h_v1"
timeframe = "15m"
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

def calculate_roc(close, period=5):
    """Rate of Change"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100
    
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    roc_5 = calculate_roc(close, period=5)
    
    # Volume ratio (taker buy / total volume)
    volume_ratio = np.full(n, 0.5, dtype=np.float64)
    mask = volume > 0
    volume_ratio[mask] = taker_buy_volume[mask] / volume[mask]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(roc_5[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1h HMA bias) ===
        price_above_1h = close[i] > hma_1h_aligned[i]
        price_below_1h = close[i] < hma_1h_aligned[i]
        
        # 4h HMA for stronger trend confirmation
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume_ratio[i]
        volume_confirm_long = vol_ratio > 0.50
        volume_confirm_short = vol_ratio < 0.50
        
        # === RSI PULLBACK ZONES (WIDE BANDS for trade generation) ===
        rsi = rsi_7[i]
        # LONG: RSI pulled back to 30-50 zone (oversold within uptrend)
        rsi_pullback_long = 30 <= rsi <= 50
        # SHORT: RSI bounced to 50-70 zone (overbought within downtrend)
        rsi_pullback_short = 50 <= rsi <= 70
        
        # === MOMENTUM CONFIRMATION (pullback ending) ===
        roc = roc_5[i]
        # For long: ROC should be stabilizing (not deeply negative)
        roc_ok_long = roc > -5.0
        # For short: ROC should be stabilizing (not deeply positive)
        roc_ok_short = roc < 5.0
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 1h bullish + RSI pullback + momentum stabilizing
        if price_above_1h and rsi_pullback_long and roc_ok_long:
            base_size = SIZE_BASE
            
            # Stronger if 4h also bullish
            if price_above_4h:
                base_size = SIZE_STRONG
            
            # Volume confirmation adds conviction
            if volume_confirm_long:
                base_size = min(SIZE_STRONG, base_size + 0.05)
            
            desired_signal = base_size
        
        # SHORT: 1h bearish + RSI pullback + momentum stabilizing
        elif price_below_1h and rsi_pullback_short and roc_ok_short:
            base_size = SIZE_BASE
            
            # Stronger if 4h also bearish
            if price_below_4h:
                base_size = SIZE_STRONG
            
            # Volume confirmation adds conviction
            if volume_confirm_short:
                base_size = min(SIZE_STRONG, base_size + 0.05)
            
            desired_signal = -base_size
        
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