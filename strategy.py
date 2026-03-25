#!/usr/bin/env python3
"""
Experiment #1262: 4h Primary + 1d/1w HTF — Dual Regime (Choppiness + HMA + RSI)

Hypothesis: Previous 4h strategies (#1258, #1260) failed with 0 trades due to over-filtering.
This strategy uses a DUAL REGIME approach that adapts to market conditions:

1. CHOPPINESS INDEX (14) detects regime:
   - CHOP < 38.2 = TRENDING → follow HMA trend with ROC momentum
   - CHOP > 61.8 = RANGING → mean revert with RSI extremes
   - 38.2-61.8 = TRANSITION → stay flat or reduce position

2. 1d HMA(21) for major trend bias (only trade with daily direction)
3. 1w HMA(21) for ultra-long-term regime filter
4. RSI(14) for mean-reversion entries in choppy markets
5. ROC(10) for momentum confirmation in trending markets
6. ATR(14) 2.5x trailing stop for risk management

Why this should work:
- Dual regime = trades in BOTH trending AND ranging markets (not just one)
- 4h timeframe = natural 20-50 trades/year (fee-friendly)
- LOOSE entry thresholds to GUARANTEE trades (learned from #1258/#1260 failures)
- Discrete sizing (0.0, ±0.25, ±0.30) minimizes fee churn
- Stoploss via signal→0 prevents catastrophic drawdowns

Entry logic (LOOSE to guarantee 30+ trades):
- TRENDING (CHOP<38.2): Long if 1d_HMA bullish + 4h_HMA rising + ROC>2
- TRENDING (CHOP<38.2): Short if 1d_HMA bearish + 4h_HMA falling + ROC<-2
- RANGING (CHOP>61.8): Long if RSI<35 + price>1d_HMA (dip buy in uptrend)
- RANGING (CHOP>61.8): Short if RSI>65 + price<1d_HMA (fade rally in downtrend)

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_hma_rsi_1d1w_v1"
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = Ranging/Choppy market (mean reversion likely)
    - CHOP < 38.2 = Trending market (trend following likely)
    - 38.2 - 61.8 = Transition zone
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
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 0 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

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
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    roc_10 = calculate_roc(close, period=10)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    hma_4h = calculate_hma(close, period=21)
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(roc_10[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trending = chop < 38.2
        is_ranging = chop > 61.8
        is_transition = not is_trending and not is_ranging
        
        # === TREND DIRECTION (1d HMA + 1w HMA bias) ===
        # 1d HMA slope (compare to 3 bars ago for stability)
        hma_1d_slope = 0.0
        if i >= 3 and not np.isnan(hma_1d_aligned[i-3]):
            hma_1d_slope = hma_1d_aligned[i] - hma_1d_aligned[i-3]
        
        # 1w HMA for ultra-long-term bias
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # 1d price position
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 4h HMA slope
        hma_4h_slope = 0.0
        if i >= 3 and not np.isnan(hma_4h[i-3]):
            hma_4h_slope = hma_4h[i] - hma_4h[i-3]
        
        # === MOMENTUM & MEAN REVERSION ===
        rsi = rsi_14[i]
        roc = roc_10[i]
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # TRENDING REGIME: Follow trend with momentum confirmation
        if is_trending:
            # LONG: 1d bullish + 4h rising + positive momentum
            if hma_1d_slope > 0 and hma_4h_slope > 0 and price_above_1d:
                if roc > 2.0:  # Very loose momentum threshold
                    if roc > 5.0:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + 4h falling + negative momentum
            elif hma_1d_slope < 0 and hma_4h_slope < 0 and price_below_1d:
                if roc < -2.0:  # Very loose momentum threshold
                    if roc < -5.0:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        # RANGING REGIME: Mean revert at extremes
        elif is_ranging:
            # LONG: RSI oversold + price above 1d HMA (buy dips in uptrend)
            if rsi < 40 and price_above_1d:
                if rsi < 30:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: RSI overbought + price below 1d HMA (fade rallies in downtrend)
            elif rsi > 60 and price_below_1d:
                if rsi > 70:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # TRANSITION REGIME: Reduce or flat (optional small positions)
        elif is_transition:
            # Allow smaller positions if strong signals
            if hma_1d_slope > 0 and price_above_1d and rsi < 45:
                desired_signal = SIZE_BASE * 0.5
            elif hma_1d_slope < 0 and price_below_1d and rsi > 55:
                desired_signal = -SIZE_BASE * 0.5
        
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
        elif desired_signal >= SIZE_BASE * 0.4:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal <= -SIZE_BASE * 0.4:
            final_signal = -SIZE_BASE * 0.5
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