#!/usr/bin/env python3
"""
Experiment #1200: 6h Primary + 1d/1w HTF — Fisher Transform + HMA Trend + Volume

Hypothesis: After 990+ failed experiments, the key insight is:
1. Fisher Transform catches reversals better than RSI in bear/range markets (research-backed)
2. 6h timeframe is underexplored (ZERO prior experiments before #1191)
3. Volume confirmation filters false breakouts without killing trade frequency
4. Simple 1d HMA trend filter works (proven in current best 6h strategy)

Why this is DIFFERENT from failed attempts:
- NOT using weekly pivots (failed 10+ times: #1191, woodie_pivot, weekly_pivot_*)
- NOT using choppiness index (causes Sharpe=0.000 in #1190, #1195, #1199)
- NOT using complex regime switches (causes 0 trades repeatedly)
- Uses Fisher Transform instead of RSI (different signal type, better for reversals)
- Adds volume confirmation (taker_buy_volume ratio - new filter for 6h)

Entry logic (LOOSE to guarantee >=30 trades/year):
- LONG: price > 1d_HMA AND Fisher crosses above -1.5 AND volume ratio > 0.40
- SHORT: price < 1d_HMA AND Fisher crosses below +1.5 AND volume ratio < 0.60
- Weekly HMA for confirmation (increases size when aligned)

Fisher Transform thresholds (from research):
- Cross above -1.5 = oversold reversal signal
- Cross below +1.5 = overbought reversal signal
- These trigger frequently enough for 30-60 trades/year on 6h

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_volume_1d1w_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - catches reversals in bear/range markets
    Reference: Ehlers, J.F. (2002) "Fisher Transform"
    
    Steps:
    1. Calculate typical price: (high + low + close) / 3
    2. Normalize to -1 to +1 range using highest high / lowest low over period
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    4. Smooth with EMA
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low + close) / 3.0
    
    # Normalize to -1 to +1
    normalized = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        if highest > lowest:
            normalized[i] = 0.999 * (2.0 * (typical[i] - lowest) / (highest - lowest) - 1.0)
            # Clamp to prevent log errors
            normalized[i] = np.clip(normalized[i], -0.999, 0.999)
    
    # Fisher transform
    fisher = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(normalized[i]):
            fisher[i] = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i]))
    
    # Smooth fisher with EMA
    fisher_smooth = pd.Series(fisher).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher, fisher_smooth

def calculate_volume_ratio(taker_buy_volume, volume):
    """Taker buy volume ratio (0.0 to 1.0)"""
    ratio = np.divide(taker_buy_volume, volume, out=np.zeros_like(taker_buy_volume), where=volume != 0)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
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
    fisher_raw, fisher_smooth = calculate_fisher_transform(high, low, close, period=9)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    
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
    
    # Track previous fisher for crossover detection
    prev_fisher = np.nan
    
    # Warmup period
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher_smooth[i] if not np.isnan(fisher_smooth[i]) else prev_fisher
            continue
        
        if np.isnan(fisher_smooth[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher_smooth[i] if not np.isnan(fisher_smooth[i]) else prev_fisher
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher_smooth[i] if not np.isnan(fisher_smooth[i]) else prev_fisher
            continue
        
        # === TREND DIRECTION (Daily HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # Weekly HMA for additional confirmation
        hma_1w_valid = not np.isnan(hma_1w_aligned[i])
        price_above_1w = hma_1w_valid and close[i] > hma_1w_aligned[i]
        price_below_1w = hma_1w_valid and close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher = fisher_smooth[i]
        fisher_cross_up = not np.isnan(prev_fisher) and prev_fisher < -1.5 and fisher >= -1.5
        fisher_cross_down = not np.isnan(prev_fisher) and prev_fisher > 1.5 and fisher <= 1.5
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = vol_ratio[i]
        vol_bullish = vol_ratio > 0.40  # More buying pressure
        vol_bearish = vol_ratio < 0.60  # More selling pressure
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: Price above 1d HMA + Fisher cross above -1.5 + Volume confirmation
        if price_above_1d and fisher_cross_up and vol_bullish:
            if price_above_1w:
                desired_signal = SIZE_STRONG  # Strong trend alignment
            else:
                desired_signal = SIZE_BASE  # Basic uptrend
        
        # SHORT: Price below 1d HMA + Fisher cross below +1.5 + Volume confirmation
        elif price_below_1d and fisher_cross_down and vol_bearish:
            if price_below_1w:
                desired_signal = -SIZE_STRONG  # Strong trend alignment
            else:
                desired_signal = -SIZE_BASE  # Basic downtrend
        
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
        prev_fisher = fisher
    
    return signals