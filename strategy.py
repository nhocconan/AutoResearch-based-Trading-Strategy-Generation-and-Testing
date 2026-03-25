#!/usr/bin/env python3
"""
Experiment #1455: 6h Primary + 12h/1d HTF — Fisher Transform + ADX Regime

Hypothesis: 6h timeframe offers optimal balance between trade frequency (30-60/year)
and signal quality. Fisher Transform excels at catching reversals in crypto's
volatile markets, while ADX filters out choppy periods. 1d HMA provides major
trend bias to avoid counter-trend disasters.

Key components:
1. 1d HMA(21) for major trend direction (long-only when bullish, short-only when bearish)
2. 6h Fisher Transform(9) for reversal entries (crosses -1.5/+1.5)
3. 6h ADX(14) > 20 for trend strength confirmation (LOOSE threshold)
4. Volume spike > 1.3x 20-bar SMA for breakout validation
5. ATR(14) trailing stop at 2.5x for risk management
6. Mean-reversion fallback when ADX < 20 (RSI extremes)
7. Discrete position sizing: 0.0, ±0.20, ±0.30

Why this should work:
- Fisher Transform normalizes price to Gaussian distribution, better reversal signals
- ADX filter prevents entries during chop (major failure mode of prior strategies)
- 1d HMA bias prevents major counter-trend losses (2022 crash protection)
- Volume confirmation reduces false breakouts
- Mean-reversion fallback ensures trades even in low-ADX periods
- 6h TF = ~40 trades/year (fee-efficient)

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_adx_volume_1d_v1"
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        plus_dm[i] = max(0, high[i] - high[i-1]) if high[i] - high[i-1] > low[i-1] - low[i] else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if low[i-1] - low[i] > high[i] - high[i-1] else 0
    
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    plus_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = tr_smooth > 0
    plus_di[mask] = 100 * plus_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100 * minus_smooth[mask] / tr_smooth[mask]
    
    dx = np.full(n, np.nan)
    mask2 = (plus_di + minus_di) > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / (plus_di[mask2] + minus_di[mask2])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_fisher(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Better for catching reversals than RSI in volatile markets
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low + close) / 3.0
    
    # Normalize to -1 to +1 range
    fisher_input = np.full(n, np.nan, dtype=np.float64)
    fisher_input[period-1] = 0.0  # Initialize
    
    for i in range(period, n):
        highest = np.max(typical[i - period + 1:i + 1])
        lowest = np.min(typical[i - period + 1:i + 1])
        range_val = highest - lowest
        if range_val > 1e-10:
            fisher_input[i] = 0.66 * ((typical[i] - lowest) / range_val - 0.5) + 0.67 * fisher_input[i-1]
            fisher_input[i] = np.clip(fisher_input[i], -0.99, 0.99)
    
    # Fisher transform
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        if not np.isnan(fisher_input[i]) and abs(fisher_input[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1 + fisher_input[i]) / (1 - fisher_input[i]))
            trigger[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else fisher[i]
    
    return fisher, trigger

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_21 = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    fisher, fisher_trigger = calculate_fisher(high, low, close, period=9)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume SMA for spike detection
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma20[i]) or vol_sma20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        adx = adx_14[i]
        is_trending = adx > 20  # LOOSE threshold to ensure trades
        is_choppy = adx < 20
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_val = fisher[i]
        fisher_prev = fisher_trigger[i]  # previous bar's fisher
        
        # Fisher long: crosses above -1.5 from below
        fisher_long = fisher_val > -1.5 and fisher_prev <= -1.5
        
        # Fisher short: crosses below +1.5 from above
        fisher_short = fisher_val < 1.5 and fisher_prev >= 1.5
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / vol_sma20[i]
        vol_spike = vol_ratio > 1.3  # 30% above average
        
        # === HMA CONFIRMATION ===
        hma_bullish = close[i] > hma_21[i]
        hma_bearish = close[i] < hma_21[i]
        
        # === RSI MEAN REVERSION (fallback when ADX low) ===
        rsi = rsi_14[i]
        rsi_oversold = rsi < 35
        rsi_overbought = rsi > 65
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND MODE (ADX > 20): Fisher reversals with trend bias
        if is_trending:
            # LONG: 1d bullish + Fisher long + HMA bullish OR volume spike
            if price_above_1d and fisher_long:
                if hma_bullish or vol_spike:
                    desired_signal = SIZE_STRONG if hma_bullish else SIZE_BASE
            
            # SHORT: 1d bearish + Fisher short + HMA bearish OR volume spike
            elif price_below_1d and fisher_short:
                if hma_bearish or vol_spike:
                    desired_signal = -SIZE_STRONG if hma_bearish else -SIZE_BASE
        
        # CHOPPY MODE (ADX < 20): RSI mean reversion with 1d bias
        elif is_choppy:
            # LONG: 1d bullish + RSI oversold
            if price_above_1d and rsi_oversold:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + RSI overbought
            elif price_below_1d and rsi_overbought:
                desired_signal = -SIZE_BASE
        
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