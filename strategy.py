#!/usr/bin/env python3
"""
Experiment #1098: 4h Primary + 1d HTF — Volatility Regime + Fisher Transform + HMA Trend

Hypothesis: Volatility-based regime detection (ATR ratio) combined with Ehlers Fisher Transform
for entry timing will outperform RSI-based strategies, especially in bear/range markets like 2022-2023.

Key innovations:
1. Volatility Regime: ATR(7)/ATR(30) ratio
   - >2.0 = vol spike (mean reversion opportunity after panic)
   - <1.2 = vol crush (trend continuation likely)
   - 1.2-2.0 = normal (use trend following)
2. Ehlers Fisher Transform (period=9): Better reversal detection than RSI in bear markets
   - Long when Fisher crosses above -1.5 from below
   - Short when Fisher crosses below +1.5 from above
3. HMA(21) on 1d for trend bias - only take Fisher signals in trend direction
4. ATR-based position sizing: reduce size when vol is extreme
5. Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn

Why this should work:
- Fisher Transform normalizes price action, works better than RSI in non-stationary markets
- Vol regime detection captures panic/recovery cycles (2022 crash had multiple vol spikes)
- 4h timeframe = 20-50 trades/year target (fee drag manageable)
- HMA filter prevents counter-trend trades in strong moves
- ATR sizing reduces exposure when vol is dangerous

Entry conditions (LOOSE to guarantee trades):
- LONG vol spike: ATR_ratio>1.8 + Fisher<-1.0 + price>1d_HMA*0.92
- LONG trend: ATR_ratio<1.5 + Fisher cross above -1.0 + price>1d_HMA
- SHORT vol spike: ATR_ratio>1.8 + Fisher>1.0 + price<1d_HMA*1.08
- SHORT trend: ATR_ratio<1.5 + Fisher cross below +1.0 + price<1d_HMA

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 4h
Size: 0.20-0.30 discrete (volatility adaptive)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_fisher_hma_regime_1d_v1"
timeframe = "4h"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to -1 to +1 range
    Better reversal detection than RSI in non-stationary markets
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest)
    3. Transform: 0.5 * ln((1+x)/(1-x)) where x = 2*normalized - 1
    """
    n = len(close)
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Get highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        # Normalize price to 0-1 range
        typical_price = (high[i] + low[i]) / 2.0
        normalized = (typical_price - lowest) / price_range
        
        # Transform to -1 to +1
        x = 2.0 * normalized - 1.0
        x = np.clip(x, -0.999, 0.999)  # Prevent log(0)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Signal line (1-period lag of fisher)
        if i > period:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_vol_ratio(atr_short, atr_long):
    """ATR ratio for volatility regime detection"""
    n = len(atr_short)
    ratio = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(n):
        if not np.isnan(atr_short[i]) and not np.isnan(atr_long[i]) and atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

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
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    vol_ratio = calculate_vol_ratio(atr_7, atr_30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    
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
    
    # Track Fisher crosses
    prev_fisher = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(vol_ratio[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        # === VOLATILITY REGIME DETECTION ===
        is_vol_spike = vol_ratio[i] > 1.8  # High vol = mean reversion opportunity
        is_vol_crush = vol_ratio[i] < 1.3  # Low vol = trend continuation
        # 1.3-1.8 = normal
        
        # === HTF TREND BIAS ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = False
        fisher_cross_down = False
        
        if not np.isnan(prev_fisher) and not np.isnan(fisher_signal[i]):
            # Cross above -1.5 (bullish reversal)
            if prev_fisher < -1.3 and fisher[i] > -1.3:
                fisher_cross_up = True
            # Cross below +1.5 (bearish reversal)
            if prev_fisher > 1.3 and fisher[i] < 1.3:
                fisher_cross_down = True
        
        prev_fisher = fisher[i]
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        # VOL SPIKE REGIME - Mean reversion after panic
        if is_vol_spike:
            # Long: Fisher oversold + price near or above 1d HMA
            if fisher[i] < -1.0 and close[i] > hma_1d_aligned[i] * 0.95:
                desired_signal = SIZE_BASE
            # Strong long: Fisher very oversold
            elif fisher[i] < -1.5 and close[i] > hma_1d_aligned[i] * 0.90:
                desired_signal = SIZE_STRONG
            # Short: Fisher overbought + price near or below 1d HMA
            elif fisher[i] > 1.0 and close[i] < hma_1d_aligned[i] * 1.05:
                desired_signal = -SIZE_BASE
            # Strong short: Fisher very overbought
            elif fisher[i] > 1.5 and close[i] < hma_1d_aligned[i] * 1.10:
                desired_signal = -SIZE_STRONG
        
        # VOL CRUSH / NORMAL REGIME - Trend following
        else:
            # Long: Bullish trend + Fisher cross up or Fisher > -0.5
            if hma_1d_bull:
                if fisher_cross_up:
                    desired_signal = SIZE_STRONG
                elif fisher[i] > -0.5 and fisher[i] < 1.0:
                    desired_signal = SIZE_BASE
            # Short: Bearish trend + Fisher cross down or Fisher < +0.5
            elif hma_1d_bear:
                if fisher_cross_down:
                    desired_signal = -SIZE_STRONG
                elif fisher[i] < 0.5 and fisher[i] > -1.0:
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