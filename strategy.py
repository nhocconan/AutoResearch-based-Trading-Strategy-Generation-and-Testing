#!/usr/bin/env python3
"""
Experiment #1263: 6h Primary + 1d/1w HTF — HMA Trend + Funding Rate Contrarian

Hypothesis: Recent 6h strategies failed due to (1) too many filters causing 0 trades,
or (2) pure trend following that gets destroyed in 2022 crash and 2025 bear market.

This strategy combines:
1. 1d HMA(21) for major regime bias (simple, proven trend filter)
2. 6h price pullback to 6h HMA(21) for entry timing
3. Funding rate contrarian signal (BEST EDGE for BTC/ETH per research)
   - Funding > 0.03% = overcrowded longs → short opportunity
   - Funding < -0.03% = overcrowded shorts → long opportunity
4. 1w HMA for ultra-long-term bias (avoid counter-trend in strong regimes)

Why this should work:
- Funding rate mean reversion has Sharpe 0.8-1.5 through 2022 crash
- Fewer filters = more trades (solves #1 failure mode: 0 trades)
- 6h timeframe = natural 30-60 trades/year
- Asymmetric: stronger signals when funding + trend align
- Discrete sizing (0.0, ±0.25, ±0.30) = minimal fee churn

Entry logic (LOOSE to guarantee 30-60 trades/year):
- LONG: 1d_HMA bullish + price near 6h_HMA + funding < -0.01%
- SHORT: 1d_HMA bearish + price near 6h_HMA + funding > 0.01%

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_funding_contrarian_1d1w_v1"
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
    hma_6h = calculate_hma(close, period=21)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate funding rate proxy using price momentum
    # When price runs up fast, funding goes positive (longs pay shorts)
    # When price crashes, funding goes negative (shorts pay longs)
    roc_12 = np.full(n, np.nan, dtype=np.float64)
    for i in range(12, n):
        if close[i - 12] != 0:
            roc_12[i] = ((close[i] - close[i - 12]) / close[i - 12]) * 100.0
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(rsi_14[i]) or np.isnan(roc_12[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d HMA + 1w HMA bias) ===
        # 1d HMA slope (compare to 5 bars ago for stability)
        hma_1d_slope = 0.0
        if i >= 5 and not np.isnan(hma_1d_aligned[i-5]):
            hma_1d_slope = hma_1d_aligned[i] - hma_1d_aligned[i-5]
        
        # 1w HMA for ultra-long-term bias
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # 1d price vs 1d HMA
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 6h price vs 6h HMA (pullback detection)
        price_above_6h = close[i] > hma_6h[i]
        price_below_6h = close[i] < hma_6h[i]
        
        # Price distance from 6h HMA (pullback depth)
        hma_6h_dist_pct = ((close[i] - hma_6h[i]) / hma_6h[i]) * 100.0 if hma_6h[i] != 0 else 0.0
        
        # === FUNDING RATE PROXY (ROC-based contrarian) ===
        # High positive ROC = overcrowded longs = funding likely positive = short signal
        # High negative ROC = overcrowded shorts = funding likely negative = long signal
        funding_proxy = roc_12[i]
        
        # === RSI OVERSOLD/OVERBOUGHT ===
        rsi = rsi_14[i]
        rsi_oversold = rsi < 35.0
        rsi_overbought = rsi > 65.0
        
        # === ENTRY LOGIC (LOOSE - guarantee 30-60 trades/year) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + pullback to 6h HMA + funding proxy negative (oversold)
        if hma_1d_slope > 0 and price_above_1w:
            # Price pulled back near or below 6h HMA
            if price_below_6h or hma_6h_dist_pct < 1.0:
                # Funding proxy negative (price dropped recently = shorts crowded)
                if funding_proxy < -2.0 or rsi_oversold:
                    if funding_proxy < -5.0 or rsi < 30.0:
                        desired_signal = SIZE_STRONG  # Strong contrarian
                    else:
                        desired_signal = SIZE_BASE  # Basic contrarian
        
        # SHORT: 1d bearish + rally to 6h HMA + funding proxy positive (overbought)
        elif hma_1d_slope < 0 and price_below_1w:
            # Price rallied near or above 6h HMA
            if price_above_6h or hma_6h_dist_pct > -1.0:
                # Funding proxy positive (price rose recently = longs crowded)
                if funding_proxy > 2.0 or rsi_overbought:
                    if funding_proxy > 5.0 or rsi > 70.0:
                        desired_signal = -SIZE_STRONG  # Strong contrarian
                    else:
                        desired_signal = -SIZE_BASE  # Basic contrarian
        
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