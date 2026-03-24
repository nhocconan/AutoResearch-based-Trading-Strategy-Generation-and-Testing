#!/usr/bin/env python3
"""
Experiment #1023: 6h Primary + 1d/1w HTF — Vol Spike Reversion + HMA Trend + RSI Pullback

Hypothesis: 6h timeframe captures multi-day swings without noise. Combining vol spike
detection (ATR ratio) with HTF trend bias and RSI pullback entries will work better
than pure trend-following in bear/range markets (2022-2023, 2025+).

Key innovations:
1. Vol Spike Detection: ATR(7)/ATR(30) > 2.0 indicates panic/extreme vol
2. Vol Crush Entry: After vol spike, enter when ATR ratio drops < 1.3 + RSI extreme
3. HTF Trend Bias: 1w HMA(21) for long-term, 1d HMA(21) for intermediate
4. Regime Filter: CHOP(14) to distinguish trend vs range modes
5. RSI Pullback: In uptrend, buy RSI 35-50 pullbacks; in downtrend, sell RSI 50-65 rallies
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work:
- Vol spike reversion captures panic bottoms (2022 crash, 2025 drawdown)
- HTF bias prevents counter-trend trades in strong trends
- 6h TF balances trade frequency (30-60/year) with signal quality
- Simpler than failed weekly pivot strategies (more trades guaranteed)
- Regime-adaptive: mean revert in chop, trend-follow in clean trends

Entry conditions (LOOSE to guarantee trades):
- LONG vol-revert: ATR_ratio>2.0 then <1.3 + RSI<35 + price>1w_HMA*0.90
- LONG trend-pullback: price>1d_HMA>1w_HMA + RSI 35-50 + CHOP<50
- SHORT vol-revert: ATR_ratio>2.0 then <1.3 + RSI>65 + price<1w_HMA*1.10
- SHORT trend-pullback: price<1d_HMA<1w_HMA + RSI 50-65 + CHOP<50

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_volspike_hma_rsi_regime_1d1w_v1"
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
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR ratio for vol spike detection"""
    n = len(close)
    if n < long_period + 1:
        return np.full(n, np.nan)
    
    atr_short = calculate_atr(high, low, close, period=short_period)
    atr_long = calculate_atr(high, low, close, period=long_period)
    
    ratio = np.divide(atr_short, atr_long, out=np.full(n, np.nan), where=atr_long > 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    
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
    
    # Vol spike tracking
    vol_spike_detected = False
    vol_spike_bar = -100
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOL SPIKE DETECTION ===
        if atr_ratio[i] > 2.0:
            vol_spike_detected = True
            vol_spike_bar = i
        
        # Vol spike cooldown (must be > 20 bars ago to reset)
        if vol_spike_detected and (i - vol_spike_bar) > 50:
            vol_spike_detected = False
        
        vol_crush = vol_spike_detected and atr_ratio[i] < 1.3 and (i - vol_spike_bar) >= 5
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 55.0  # Range market
        is_trending = chop_14[i] < 45.0  # Trend market
        
        # === HTF BIAS (HMA alignment) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong trend alignment
        strong_bull = hma_1d_bull and hma_1w_bull and hma_1d_aligned[i] > hma_1w_aligned[i]
        strong_bear = hma_1d_bear and hma_1w_bear and hma_1d_aligned[i] < hma_1w_aligned[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # VOL SPIKE REVERSION (works in any regime)
        if vol_crush:
            # Long after vol spike + RSI oversold + above weekly HMA support
            if rsi_14[i] < 35.0 and close[i] > hma_1w_aligned[i] * 0.92:
                desired_signal = SIZE_STRONG
            # Short after vol spike + RSI overbought + below weekly HMA resistance
            elif rsi_14[i] > 65.0 and close[i] < hma_1w_aligned[i] * 1.08:
                desired_signal = -SIZE_STRONG
        
        # TREND PULLBACK (in trending regime)
        elif is_trending:
            # Long pullback in uptrend
            if strong_bull and 35.0 <= rsi_14[i] <= 52.0:
                desired_signal = SIZE_BASE
            elif hma_1d_bull and hma_1w_bull and 38.0 <= rsi_14[i] <= 50.0:
                desired_signal = SIZE_BASE
            
            # Short rally in downtrend
            elif strong_bear and 48.0 <= rsi_14[i] <= 65.0:
                desired_signal = -SIZE_BASE
            elif hma_1d_bear and hma_1w_bear and 50.0 <= rsi_14[i] <= 62.0:
                desired_signal = -SIZE_BASE
        
        # RANGE MEAN REVERSION (in choppy regime)
        elif is_choppy:
            # Long at range bottom
            if rsi_14[i] < 32.0 and close[i] > hma_1w_aligned[i] * 0.88:
                desired_signal = SIZE_BASE
            # Short at range top
            elif rsi_14[i] > 68.0 and close[i] < hma_1w_aligned[i] * 1.12:
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