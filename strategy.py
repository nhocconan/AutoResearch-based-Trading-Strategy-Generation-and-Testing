#!/usr/bin/env python3
"""
Experiment #1355: 6h Primary + 12h/1d HTF — HMA Trend + RSI Pullback + Volume Confirm

Hypothesis: 6h timeframe is ideal for swing trading - captures multi-day moves without
the noise of lower TFs. Using 12h HMA for trend direction + 6h RSI pullback entries
should generate consistent trades with positive Sharpe.

Key insight from failures:
- CRSI + CHOP on 6h failed badly (Sharpe=-3.977) - too complex
- Weekly pivot concepts failed - 1w too slow for 6h entries
- Many strategies generated 0 trades - entry conditions too strict

This strategy uses:
1. 12h HMA(21) for intermediate trend direction (faster than 1d, responsive for 6h)
2. 1d HMA(21) for major regime filter (avoid counter-trend in strong moves)
3. 6h RSI(14) pullback entries (RSI 35-50 for long, 50-65 for short)
4. 6h EMA(21) for pullback confirmation (price near EMA = good entry)
5. Volume spike filter (optional confirmation, not required)
6. ATR(14) 2.5x trailing stoploss

Why 6h works:
- 4 bars per day = captures daily moves without intraday noise
- 28 bars per week = good for weekly swing trades
- Target: 30-60 trades/year (fee-friendly)

Entry logic (LOOSE to guarantee trades):
- LONG: 12h_HMA sloping up + price > 1d_HMA + RSI(6h) 35-50 + price near EMA(6h)
- SHORT: 12h_HMA sloping down + price < 1d_HMA + RSI(6h) 50-65 + price near EMA(6h)

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_trend_rsi_pullback_12h1d_v1"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio[vol_ratio == np.inf] = np.nan
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    ema_21 = calculate_ema(close, period=21)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate 12h HMA slope (trend direction)
    hma_12h_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(hma_12h_aligned[i]) and not np.isnan(hma_12h_aligned[i-1]):
            hma_12h_slope[i] = hma_12h_aligned[i] - hma_12h_aligned[i-1]
    
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
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_21[i]) or np.isnan(hma_12h_slope[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (12h HMA slope + 1d HMA position) ===
        hma_12h_rising = hma_12h_slope[i] > 0
        hma_12h_falling = hma_12h_slope[i] < 0
        
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === PULLBACK CONFIRMATION (price near 6h EMA) ===
        # Price within 2% of EMA = good pullback entry
        ema_distance = (close[i] - ema_21[i]) / ema_21[i] if ema_21[i] != 0 else 0
        near_ema_long = ema_distance < 0.02 and ema_distance > -0.05  # Slightly below to at EMA
        near_ema_short = ema_distance > -0.02 and ema_distance < 0.05  # Slightly above to at EMA
        
        # === RSI PULLBACK (LOOSE - guarantee trades) ===
        rsi = rsi_14[i]
        rsi_pullback_long = 35 <= rsi <= 55  # Pullback in uptrend
        rsi_pullback_short = 45 <= rsi <= 65  # Pullback in downtrend
        
        # === VOLUME CONFIRMATION (OPTIONAL - boosts size) ===
        vol_confirmed = False
        if not np.isnan(vol_ratio[i]) and vol_ratio[i] > 1.2:
            vol_confirmed = True
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 12h rising + price > 1d_HMA + RSI pullback + near EMA
        if hma_12h_rising and price_above_1d and rsi_pullback_long and near_ema_long:
            if vol_confirmed:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 12h falling + price < 1d_HMA + RSI pullback + near EMA
        elif hma_12h_falling and price_below_1d and rsi_pullback_short and near_ema_short:
            if vol_confirmed:
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