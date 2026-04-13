#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w ATR regime filter and volume confirmation
    # Long: Close > H3 AND 1w ATR ratio < 0.8 (low volatility regime) AND volume > 1.5x avg
    # Short: Close < L3 AND 1w ATR ratio < 0.8 (low volatility regime) AND volume > 1.5x avg
    # Exit: Close < H3 for longs OR Close > L3 for shorts OR volatility expansion (ATR ratio > 1.2)
    # Using 1d timeframe for low trade frequency, Camarilla for intraday structure,
    # 1w ATR for volatility regime filter (avoid choppy markets), volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ATR regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly ATR(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w = np.concatenate([[np.nan], tr_1w])  # Align with original length
    
    # ATR calculation with Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[period-1:period*2-1])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period*2-1, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = wilders_smoothing(tr_1w, 14)
    
    # Calculate ATR ratio (current ATR / 20-period average ATR) for regime filter
    atr_ma_1w = np.full(len(atr_1w), np.nan)
    for i in range(34, len(atr_1w)):  # 20 + 14 for proper lookback
        atr_ma_1w[i] = np.mean(atr_1w[i-20:i])
    
    atr_ratio_1w = np.where(atr_ma_1w > 0, atr_1w / atr_ma_1w, 1.0)
    atr_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio_1w)
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # H2 = close + 0.5 * (high - low)
    # H1 = close + 0.25 * (high - low)
    # L1 = close - 0.25 * (high - low)
    # L2 = close - 0.5 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    
    # Shift high/low/close by 1 to use previous day's data
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    camarilla_range = prev_high - prev_low
    h3 = prev_close + 1.0 * camarilla_range
    l3 = prev_close - 1.0 * camarilla_range
    
    # Get daily volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_1w_aligned[i]) or np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ATR ratio < 0.8 = low volatility (trending regime)
        low_vol_regime = atr_ratio_1w_aligned[i] < 0.8
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Camarilla breakout + low volatility regime + volume confirmation
        long_entry = (close[i] > h3[i]) and low_vol_regime and vol_confirm
        short_entry = (close[i] < l3[i]) and low_vol_regime and vol_confirm
        
        # Exit logic: Close back inside H3/L3 OR volatility expansion (ATR ratio > 1.2)
        long_exit = (close[i] < h3[i]) or (atr_ratio_1w_aligned[i] > 1.2)
        short_exit = (close[i] > l3[i]) or (atr_ratio_1w_aligned[i] > 1.2)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_camarilla_pivot_breakout_v1"
timeframe = "1d"
leverage = 1.0