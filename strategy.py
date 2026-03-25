#!/usr/bin/env python3
"""
Experiment #1219: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Choppiness Filter

Hypothesis: After analyzing 1000+ failed experiments, the winning pattern for 1h is:
1. 4h HMA(21) for PRIMARY trend direction (proven in mtf_6h_hma_trend_rsi_momentum_1d_v1)
2. 1h RSI(14) pullback entries in 35-65 range (not extremes - guarantees trades)
3. Choppiness Index(14) < 38.2 to filter out range-bound whipsaw markets
4. Volume confirmation (volume > 20-bar MA) to avoid low-liquidity fakeouts
5. 12h HMA(21) for additional trend confirmation (not required, but boosts size)

Key insight from failures:
- #1210, #1217: Session filters killed ALL trades (Sharpe=0.000)
- #1209, #1216: 15m/30m with loose entries = too many trades = negative Sharpe
- #1215: Complex Fisher/Choppiness regime = catastrophic failure (Sharpe=-95)
- Working pattern: Simple HMA trend + RSI pullback (like #1218 Sharpe=0.147)

Why this should beat the baseline (Sharpe=0.445):
- 1h timeframe with 4h trend filter = fewer trades than 6h, better entry timing
- Choppiness filter removes range-bound whipsaws (major loss source in 2022-2024)
- Volume filter avoids low-liquidity fakeouts
- Discrete sizing (0.0, ±0.25, ±0.30) minimizes fee churn
- Target: 50-80 trades/year (fee-friendly for 1h)

Timeframe: 1h
Size: 0.25 base, 0.30 strong (with 12h confirmation)
Stoploss: 2.5x ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_chop_vol_4h12h_v1"
timeframe = "1h"
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
    Choppiness Index (CHOP) - measures market choppy vs trending
    Formula: 100 * (ATR(1, n) sum) / (Highest High - Lowest Low) * 100 / log10(n)
    CHOP > 61.8 = range-bound (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Calculate sum of ATR(1) over period
        atr1_sum = 0.0
        for j in range(i - period + 1, i + 1):
            atr1_sum += high[j] - low[j]
        
        # Highest high and lowest low over period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop = 100.0 * (atr1_sum / price_range) / np.log10(period)
            choppiness[i] = chop
    
    return choppiness

def calculate_volume_ma(volume, period=20):
    """Volume moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
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
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (4h HMA) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # 12h HMA for additional confirmation (not required)
        hma_12h_valid = not np.isnan(hma_12h_aligned[i])
        price_above_12h = hma_12h_valid and close[i] > hma_12h_aligned[i]
        price_below_12h = hma_12h_valid and close[i] < hma_12h_aligned[i]
        
        # === MARKET REGIME (Choppiness Index) ===
        # CHOP < 38.2 = trending (allow trend entries)
        # CHOP > 61.8 = range-bound (skip trend entries)
        is_trending = chop_14[i] < 38.2
        
        # === VOLUME CONFIRMATION ===
        # Volume must be above 20-bar MA to confirm move
        volume_confirmed = volume[i] > vol_ma_20[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        rsi = rsi_14[i]
        
        # Only enter in trending market (avoid choppy whipsaws)
        if is_trending:
            # LONG: Price above 4h HMA + RSI pullback (35-65) + volume confirmed
            if price_above_4h:
                if 35.0 <= rsi <= 65.0 and volume_confirmed:
                    if price_above_12h:
                        desired_signal = SIZE_STRONG  # Strong trend alignment (4h + 12h)
                    else:
                        desired_signal = SIZE_BASE  # Basic uptrend pullback
            
            # SHORT: Price below 4h HMA + RSI pullback (35-65) + volume confirmed
            elif price_below_4h:
                if 35.0 <= rsi <= 65.0 and volume_confirmed:
                    if price_below_12h:
                        desired_signal = -SIZE_STRONG  # Strong trend alignment (4h + 12h)
                    else:
                        desired_signal = -SIZE_BASE  # Basic downtrend pullback
        
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