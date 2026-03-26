#!/usr/bin/env python3
"""
Experiment #025: 4h Camarilla Bounce + Volume + 1d Trend + Choppiness

HYPOTHESIS: Camarilla pivot levels (S4/R4) are strong mean reversion points where
price bounces. Combined with volume confirmation, choppiness regime filter, and
1d HMA trend alignment, this catches reversals in both bull and bear markets:
- BULL: buy dips to S4 when 1d trend bullish
- BEAR: short rallies to R4 when 1d trend bearish
Choppiness filter avoids whipsaws in trending markets (CHOP > 61.8 = range = good).

DB TOP PERFORMER: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 (ETH: 1.47 Sharpe, 95 trades)

TIMEFRAME: 4h
HTF: 1d for trend
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_bounce_vol_chop_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def calculate_choppiness(high, close, period=14):
    """Choppiness Index - values > 61.8 = choppy/range, < 38.2 = trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        atr_sum = 0.0
        high_low_sum = 0.0
        valid = True
        
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
            hl_range = high[j] - low[j]
            if hl_range <= 0:
                valid = False
                break
            high_low_sum += hl_range
        
        if valid and high_low_sum > 0:
            chop[i] = 100 * (np.log10(atr_sum) / np.log10(high_low_sum))
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend (aligned)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Previous 1d OHLC for Camarilla (aligned + shifted)
    prev_close_1d = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    prev_high_1d = align_htf_to_ltf(prices, df_1d, df_1d['high'].values)
    prev_low_1d = align_htf_to_ltf(prices, df_1d, df_1d['low'].values)
    
    # Calculate 4h indicators
    atr_4h = calculate_atr(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, close, period=14)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Check HTF data availability
        hma_1d_val = hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else None
        prev_close = prev_close_1d[i] if not np.isnan(prev_close_1d[i]) else None
        prev_high = prev_high_1d[i] if not np.isnan(prev_high_1d[i]) else None
        prev_low = prev_low_1d[i] if not np.isnan(prev_low_1d[i]) else None
        
        if hma_1d_val is None or prev_close is None or prev_high is None or prev_low is None:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === Camarilla Pivots from previous 1d ===
        daily_range = prev_high - prev_low
        
        # Camarilla levels
        s4 = prev_close - 1.1 * daily_range / 2
        s3 = prev_close - 1.1 * daily_range / 4
        r4 = prev_close + 1.1 * daily_range / 2
        r3 = prev_close + 1.1 * daily_range / 4
        
        # === FILTERS ===
        vol_confirm = vol_ratio[i] > 1.25  # Volume spike
        chop_filter = chop_4h[i] > 61.8  # Choppy = mean revert works
        bullish_trend = close[i] > hma_1d_val
        bearish_trend = close[i] < hma_1d_val
        
        # Price within pivot range
        in_s4_zone = low[i] <= s4 and close[i] >= s4 * 0.998
        in_r4_zone = high[i] >= r4 and close[i] <= r4 * 1.002
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # Long: price at S4 with volume spike, bullish trend, choppy
            if in_s4_zone and vol_confirm and bullish_trend and chop_filter:
                desired_signal = SIZE
            
            # Short: price at R4 with volume spike, bearish trend, choppy
            elif in_r4_zone and vol_confirm and bearish_trend and chop_filter:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === EXIT: RSI extreme ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            if rsi[i] < 35:
                exit_triggered = True
        
        if in_position and position_side < 0:
            if rsi[i] > 65:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            # else: same direction, maintain (no churn)
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals