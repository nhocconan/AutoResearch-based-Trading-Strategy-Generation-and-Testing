#!/usr/bin/env python3
"""
Experiment #011: 6h Ichimoku TK Cross + Cloud + Volume

HYPOTHESIS: Ichimoku Cloud is a complete institutional system (trend + momentum + structure).
The TK Cross (Tenkan-Kijun) is a fast momentum signal. Cloud defines support/resistance.
Volume spike at TK Cross confirms institutional participation. 1d HMA trend alignment ensures
we only trade with higher timeframe bias. This should work in both:
- BULL: Long breakouts with price above cloud, TK cross up, volume surge
- BEAR: Short breakdowns with price below cloud, TK cross down, volume surge
- RANGE: TK crosses within cloud = no trade (filters chop)

TIMEFRAME: 6h primary
HTF: 1d for trend alignment, 1w for regime
TARGET: 75-200 total trades over 4 years (12-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_cloud_vol_1d_v1"
timeframe = "6h"
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

def calculate_ichimoku(high, low, close, t_period=9, k_period=26, span_b_period=52, displacement=26):
    """
    Ichimoku Cloud calculation
    Returns: tenkan, kijun, senkou_a, senkou_b, cloud_bullish, price_above_cloud
    """
    n = len(close)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan = np.full(n, np.nan, dtype=np.float64)
    kijun = np.full(n, np.nan, dtype=np.float64)
    senkou_a = np.full(n, np.nan, dtype=np.float64)
    senkou_b = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(t_period - 1, n):
        window_high = np.nanmax(high[i - t_period + 1:i + 1])
        window_low = np.nanmin(low[i - t_period + 1:i + 1])
        tenkan[i] = (window_high + window_low) / 2
    
    for i in range(k_period - 1, n):
        window_high = np.nanmax(high[i - k_period + 1:i + 1])
        window_low = np.nanmin(low[i - k_period + 1:i + 1])
        kijun[i] = (window_high + window_low) / 2
    
    # Senkou Span A: (Tenkan + Kijun) / 2, plotted 26 periods ahead
    for i in range(k_period - 1, n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i + displacement] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B: (52-period high + 52-period low) / 2, plotted 26 periods ahead
    for i in range(span_b_period - 1, n):
        window_high = np.nanmax(high[i - span_b_period + 1:i + 1])
        window_low = np.nanmin(low[i - span_b_period + 1:i + 1])
        senkou_b[i + displacement] = (window_high + window_low) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d HMA for trend alignment
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 1w HMA for regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # === Calculate Ichimoku ===
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for additional confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 60  # Need enough for Ichimoku (26 + displacement)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === ICHIMOKU SIGNALS ===
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        
        # TK Cross detection (need previous values)
        tenkan_prev = tenkan[i - 1] if i > 0 and not np.isnan(tenkan[i - 1]) else tenkan_val
        kijun_prev = kijun[i - 1] if i > 0 and not np.isnan(kijun[i - 1]) else kijun_val
        
        # Bullish TK Cross: Tenkan crosses above Kijun
        tk_bullish_cross = (tenkan_val > kijun_val) and (tenkan_prev <= kijun_prev)
        # Bearish TK Cross: Tenkan crosses below Kijun
        tk_bearish_cross = (tenkan_val < kijun_val) and (tenkan_prev >= kijun_prev)
        
        # Cloud boundaries (current period values)
        cloud_top = max(senkou_a[i], senkou_b[i]) if not np.isnan(senkou_a[i]) and not np.isnan(senkou_b[i]) else close[i]
        cloud_bottom = min(senkou_a[i], senkou_b[i]) if not np.isnan(senkou_a[i]) and not np.isnan(senkou_b[i]) else close[i]
        
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        price_in_cloud = not price_above_cloud and not price_below_cloud
        
        # === TREND ALIGNMENT (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === REGIME (1w HMA) - only trade with trend in bear market ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5  # Volume 50% above average
        
        # === RSI MOMENTUM ===
        rsi_val = rsi[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === BULLISH ENTRY ===
            # TK cross up + price above cloud + bullish 1d trend + volume spike
            if tk_bullish_cross and price_above_cloud and price_above_1d_hma and vol_spike:
                desired_signal = SIZE
            # Alternative: Strong momentum without cross (gap up with volume)
            elif price_above_cloud and price_above_1d_hma and vol_spike and rsi_val > 60:
                if tenkan_val > kijun_val:  # Still bullish alignment
                    desired_signal = SIZE
            
            # === BEARISH ENTRY ===
            # TK cross down + price below cloud + bearish 1d trend + volume spike
            if tk_bearish_cross and price_below_cloud and not price_above_1d_hma and vol_spike:
                desired_signal = -SIZE
            # Alternative: Strong bearish momentum
            elif price_below_cloud and not price_above_1d_hma and vol_spike and rsi_val < 40:
                if tenkan_val < kijun_val:  # Still bearish alignment
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
        
        # === EXIT: Opposite signal or RSI extreme ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: TK bearish cross OR price breaks below cloud OR RSI oversold
            if tk_bearish_cross:
                exit_triggered = True
            if price_below_cloud:
                exit_triggered = True
            if rsi_val < 35:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: TK bullish cross OR price breaks above cloud OR RSI overbought
            if tk_bullish_cross:
                exit_triggered = True
            if price_above_cloud:
                exit_triggered = True
            if rsi_val > 65:
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
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            # else: same direction - maintain (no churn)
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