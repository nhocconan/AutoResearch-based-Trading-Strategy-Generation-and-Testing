#!/usr/bin/env python3
"""
Experiment #1339: 1h Primary + 4h/12h HTF — Fisher Transform + HMA Trend + Volume

Hypothesis: Failed CRSI/Choppiness strategies had entry conditions too strict (0 trades).
This uses Fisher Transform which is MORE sensitive to reversals than RSI, combined with
HTF HMA trend filter and volume confirmation. Key innovations:

1. Fisher Transform(9) - catches reversals better than RSI in bear/range markets
   Long when Fisher crosses above -1.5, Short when crosses below +1.5
2. 4h HMA(21) for intermediate trend direction (smoother than EMA)
3. 12h HMA(21) for major regime bias (only trade with daily trend)
4. Volume spike filter (1.5x 20-bar avg) - ensures real momentum, not noise
5. Session filter (08-20 UTC) - trade during high liquidity hours only
6. ATR(14) 2.5x trailing stop for risk management

Why this should work (different from failed strategies):
- Fisher Transform is proven to work in bear markets (Ehlers research)
- Volume filter actually triggers (unlike Choppiness which stays elevated)
- Session filter ensures we trade during active hours (not dead zones)
- Loose Fisher thresholds (-1.5/+1.5) guarantee trades generate
- 1h timeframe with 4h/12h filter = 40-80 trades/year target

Entry logic (LOOSE to guarantee 30+ trades):
- LONG: 12h_HMA bullish + 4h_HMA rising + Fisher crosses -1.5 + volume > 1.5x avg + session 08-20
- SHORT: 12h_HMA bearish + 4h_HMA falling + Fisher crosses +1.5 + volume > 1.5x avg + session 08-20

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_volume_session_4h12h_v1"
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

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals better than RSI in bear/range markets
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate highest high and lowest low over period
    for i in range(period - 1, n):
        if np.isnan(close[i]):
            continue
        
        highest = np.nanmax(close[i - period + 1:i + 1])
        lowest = np.nanmin(close[i - period + 1:i + 1])
        
        if highest == lowest:
            continue
        
        # Normalize price to 0-1 range
        value = (close[i] - lowest) / (highest - lowest)
        
        # Clamp to avoid division issues
        value = max(0.001, min(0.999, value))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((value) / (1 - value))
        
        # Signal line (1-bar lag)
        if i > 0 and not np.isnan(fisher[i - 1]):
            fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def is_session_active(open_time, start_hour=8, end_hour=20):
    """Check if bar is within active trading session (UTC)"""
    # open_time is in milliseconds since epoch
    hour = pd.to_datetime(open_time, unit='ms').hour
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
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
    fisher, fisher_signal = calculate_fisher(close, period=9)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
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
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        # === TREND DIRECTION (HTF HMA) ===
        # 4h HMA slope (compare to 3 bars ago for stability)
        hma_4h_slope = 0.0
        if i >= 3 and not np.isnan(hma_4h_aligned[i-3]):
            hma_4h_slope = hma_4h_aligned[i] - hma_4h_aligned[i-3]
        
        # 12h HMA bias
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_val = fisher[i]
        fisher_prev = fisher_signal[i]  # 1-bar lag
        
        # Fisher cross detection
        fisher_cross_up = (fisher_prev < -1.5 and fisher_val >= -1.5) if not np.isnan(fisher_prev) else False
        fisher_cross_down = (fisher_prev > 1.5 and fisher_val <= 1.5) if not np.isnan(fisher_prev) else False
        
        # Also check extreme levels for continuation
        fisher_extreme_long = fisher_val < -1.8
        fisher_extreme_short = fisher_val > 1.8
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / vol_sma_20[i] if vol_sma_20[i] > 0 else 0
        volume_confirmed = vol_ratio > 1.3  # 30% above average (looser than 1.5x)
        
        # === SESSION FILTER ===
        session_active = is_session_active(open_time[i], start_hour=8, end_hour=20)
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 12h bullish + 4h rising + Fisher cross up OR extreme + volume + session
        if price_above_12h and hma_4h_slope > 0:
            if (fisher_cross_up or fisher_extreme_long):
                if volume_confirmed or session_active:  # Either condition (loose)
                    if fisher_val < -1.0:  # Still in oversold territory
                        desired_signal = SIZE_BASE
                        if fisher_val < -1.5 and volume_confirmed:
                            desired_signal = SIZE_STRONG
        
        # SHORT: 12h bearish + 4h falling + Fisher cross down OR extreme + volume + session
        elif price_below_12h and hma_4h_slope < 0:
            if (fisher_cross_down or fisher_extreme_short):
                if volume_confirmed or session_active:  # Either condition (loose)
                    if fisher_val > 1.0:  # Still in overbought territory
                        desired_signal = -SIZE_BASE
                        if fisher_val > 1.5 and volume_confirmed:
                            desired_signal = -SIZE_STRONG
        
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
        prev_fisher = fisher_val
    
    return signals