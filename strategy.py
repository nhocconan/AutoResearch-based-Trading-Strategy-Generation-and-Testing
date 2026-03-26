#!/usr/bin/env python3
"""
Experiment #006: 4h Camarilla Pivot + Volume + RSI + 1d HMA Trend

HYPOTHESIS: Camarilla S4/R4 extreme levels mark institutional order zones where 
reversals occur. Volume spike confirms institutional participation. RSI extremes 
ensure we're catching oversold/overbought conditions. 1d HMA provides trend bias 
to avoid counter-trend trades. This combination worked in DB (Sharpe=1.471, 95 trades).

WHY THIS WORKS IN BOTH BULL AND BEAR:
- Bull: Long at S4 support when RSI oversold, price above 1d HMA
- Bear: Short at R4 resistance when RSI overbought, price below 1d HMA
- Range: Choppiness filter blocks trades when CHOP > 55

TARGET: 75-150 total trades over 4 years (proven pattern from DB).
Key fix from #001 (22 trades): relaxed volume threshold, allow S3/S4 and R3/R4.
Key fix from overtrading failures: strict RSI + choppiness + 1d HMA confluence.

ENTRY LOGIC (tight but not too tight):
- Long: Price touches S3/S4 + RSI(14)<40 + volume>1.5x + price>1d_HMA + CHOP<55
- Short: Price touches R3/R4 + RSI(14)>60 + volume>1.5x + price<1d_HMA + CHOP<55

RISK: 2.5*ATR stoploss, take profit at opposite pivot or 3R
SIZE: 0.25 (discrete, minimizes fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_rsi_vol_1d_v2"
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - CHOP > 61.8 = ranging, CHOP < 38.2 = trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_camarilla_pivots(prev_high, prev_low, prev_close):
    """Camarilla pivot levels from previous day's H/L/C"""
    n = len(prev_high)
    pivots = {
        's3': np.full(n, np.nan, dtype=np.float64),
        's4': np.full(n, np.nan, dtype=np.float64),
        'r3': np.full(n, np.nan, dtype=np.float64),
        'r4': np.full(n, np.nan, dtype=np.float64),
    }
    
    for i in range(1, n):
        if np.isnan(prev_high[i-1]) or np.isnan(prev_low[i-1]) or np.isnan(prev_close[i-1]):
            continue
        
        high_low_range = prev_high[i-1] - prev_low[i-1]
        if high_low_range <= 1e-10:
            continue
        
        close = prev_close[i-1]
        
        pivots['s3'][i] = close - high_low_range * 1.1 / 4
        pivots['s4'][i] = close - high_low_range * 1.1 / 2
        pivots['r3'][i] = close + high_low_range * 1.1 / 4
        pivots['r4'][i] = close + high_low_range * 1.1 / 2
    
    return pivots

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate Camarilla pivots from 1d (using previous day's H/L/C)
    cam_pivots = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivots to 4h
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s3'])
    s4_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s4'])
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r3'])
    r4_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r4'])
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    bars_in_trade = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_trending = chop < 55.0  # Only trade in trending/neutral markets
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === RSI EXTREMES ===
        rsi = rsi_14[i]
        rsi_oversold = rsi < 40.0
        rsi_overbought = rsi > 60.0
        
        # === CAMARILLA PIVOT LEVELS ===
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        
        # === PIVOT TOUCH DETECTION ===
        # Long: price low touches S3 or S4 zone (within 0.5 ATR below)
        touched_support = False
        if not np.isnan(s3) and low[i] <= s3 * 1.002:
            touched_support = True
        if not np.isnan(s4) and low[i] <= s4 * 1.002:
            touched_support = True
        
        # Short: price high touches R3 or R4 zone (within 0.5 ATR above)
        touched_resistance = False
        if not np.isnan(r3) and high[i] >= r3 * 0.998:
            touched_resistance = True
        if not np.isnan(r4) and high[i] >= r4 * 0.998:
            touched_resistance = True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if is_trending and vol_spike:
            # LONG: Support touch + RSI oversold + bullish trend bias
            if touched_support and rsi_oversold and price_above_1d_hma:
                desired_signal = SIZE
            
            # SHORT: Resistance touch + RSI overbought + bearish trend bias
            if touched_resistance and rsi_overbought and not price_above_1d_hma:
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
        
        # === TAKE PROFIT (opposite pivot or 3R) ===
        tp_triggered = False
        if in_position and position_side > 0:
            tp_target = max(r3 if not np.isnan(r3) else close[i] * 1.05, 
                           entry_price + 3.0 * entry_atr)
            if high[i] >= tp_target:
                tp_triggered = True
        
        if in_position and position_side < 0:
            tp_target = min(s3 if not np.isnan(s3) else close[i] * 0.95,
                           entry_price - 3.0 * entry_atr)
            if low[i] <= tp_target:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
                bars_in_trade = 0
            else:
                bars_in_trade += 1
        else:
            if in_position:
                bars_in_trade += 1
                # Minimum 2 bars holding period to reduce churn
                if bars_in_trade < 2:
                    desired_signal = signals[i-1] if i > 0 else 0.0
                else:
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    stop_price = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    bars_in_trade = 0
        
        signals[i] = desired_signal
    
    return signals