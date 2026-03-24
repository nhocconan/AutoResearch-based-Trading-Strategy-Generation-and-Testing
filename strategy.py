#!/usr/bin/env python3
"""
Experiment #1020: 6h Primary + 1d/1w HTF — Ehlers Fisher Transform + KAMA Adaptive Trend

Hypothesis: Ehlers Fisher Transform excels at detecting reversals in non-Gaussian crypto
distributions (unlike RSI which assumes normal distribution). Combined with KAMA (which
adapts smoothing based on market efficiency), this should catch bear market rallies and
trend reversals better than standard indicators.

Key innovations:
1. Ehlers Fisher Transform (period=9): Normalizes prices to Gaussian -1 to +1 range
   - Long when Fisher crosses above -1.5 from below (oversold reversal)
   - Short when Fisher crosses below +1.5 from above (overbought reversal)
2. KAMA (Efficiency Ratio=10): Adapts smoothing based on trend strength
   - Fast smoothing in trends, slow in chop (reduces whipsaws)
3. Volume confirmation: Entry requires volume > 20-period MA volume
4. 1d/1w HTF bias: Only long if 1d_HMA > 1w_HMA, only short if 1d_HMA < 1w_HMA
5. ATR(14) 2.5x trailing stop for risk management

Why this should work on 6h:
- Fisher Transform specifically designed for non-Gaussian distributions (crypto fits)
- KAMA reduces lag in trends while smoothing chop (perfect for 6h multi-day swings)
- 6h captures swing moves without 15m/1h noise (target 30-50 trades/year)
- HTF filter ensures we trade with higher timeframe momentum
- Volume confirmation avoids false breakouts

Entry conditions (balanced for trades):
- LONG: Fisher crosses -1.5 + price>KAMA + volume>vol_ma + 1d_HMA>1w_HMA
- SHORT: Fisher crosses +1.5 + price<KAMA + volume>vol_ma + 1d_HMA<1w_HMA

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_kama_vol_htf_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes prices to Gaussian distribution
    Reference: John Ehlers, "Rocket Science for Traders" (2002)
    
    Formula:
    1. Calculate typical price: (High + Low) / 2
    2. Normalize: (Price - Lowest) / (Highest - Lowest) with bounds 0.001 to 0.999
    3. Transform: 0.5 * ln((1+x)/(1-x)) where x is normalized price
    
    Output ranges approximately -1.5 to +1.5
    Long signal: Fisher crosses above -1.5 from below
    Short signal: Fisher crosses below +1.5 from above
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Calculate typical price
        typical = (high[i-period+1:i+1] + low[i-period+1:i+1]) / 2.0
        
        # Find highest and lowest over period
        highest = np.max(typical)
        lowest = np.min(typical)
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        # Normalize current typical price
        current_typical = (high[i] + low[i]) / 2.0
        normalized = (current_typical - lowest) / price_range
        
        # Bound to prevent division by zero in log
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Previous fisher for signal line
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts smoothing based on market efficiency
    Reference: Perry Kaufman, "Trading Systems and Methods" (1998)
    
    Efficiency Ratio (ER) = |Price Change| / Sum of |Individual Changes|
    ER near 1 = strong trend (use fast smoothing)
    ER near 0 = choppy market (use slow smoothing)
    
    Smoothing Constant = (ER * (fast_sc - slow_sc) + slow_sc)^2
    where fast_sc = 2/(fast_period+1), slow_sc = 2/(slow_period+1)
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i-er_period])
        noise = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        
        if noise < 1e-10:
            er = 1.0
        else:
            er = price_change / noise
        
        # Calculate smoothing constants
        fast_sc = 2.0 / (fast_period + 1.0)
        slow_sc = 2.0 / (slow_period + 1.0)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Initialize KAMA
        if i == er_period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

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

def calculate_volume_ma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 1e-10:
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
        
        # === HTF BIAS (1d vs 1w HMA alignment) ===
        htf_bullish = hma_1d_aligned[i] > hma_1w_aligned[i]
        htf_bearish = hma_1d_aligned[i] < hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5 and fisher_signal[i] >= -1.5
        fisher_overbought = fisher[i] > 1.5 and fisher_signal[i] <= 1.5
        
        # Fisher crossing up from oversold (long signal)
        fisher_cross_up = fisher[i] > fisher_signal[i] and fisher_signal[i] < -1.0
        # Fisher crossing down from overbought (short signal)
        fisher_cross_down = fisher[i] < fisher_signal[i] and fisher_signal[i] > 1.0
        
        # === KAMA TREND FILTER ===
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Fisher reversal + KAMA trend + HTF bullish + volume
        if fisher_cross_up and kama_bullish and htf_bullish and volume_confirmed:
            # Stronger signal if Fisher was deeply oversold
            if fisher_signal[i] < -1.5:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: Fisher reversal + KAMA trend + HTF bearish + volume
        elif fisher_cross_down and kama_bearish and htf_bearish and volume_confirmed:
            # Stronger signal if Fisher was deeply overbought
            if fisher_signal[i] > 1.5:
                desired_signal = -SIZE_STRONG
            else:
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