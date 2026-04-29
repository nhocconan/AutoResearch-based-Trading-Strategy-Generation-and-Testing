#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trending vs ranging markets
# In trending markets (JAW > TEETH > LIPS for uptrend, reverse for downtrend), we trade breakouts
# 1d EMA50 provides strong HTF trend filter to avoid counter-trend trades
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Designed for low trade frequency (target: 12-37 trades/year) on 6h timeframe to minimize fee drag
# Works in bull markets via long signals when Alligator aligned up + price > TEETH + HTF uptrend
# Works in bear markets via short signals when Alligator aligned down + price < TEETH + HTF downtrend
# Alligator's smoothed moving averages reduce whipsaw in choppy markets

name = "6h_Williams_Alligator_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: Smoothed Moving Average (SMA-like but with different smoothing)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA formula: SMMA_t = (SMMA_{t-1} * (period-1) + close_t) / period
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = 50  # warmup for Alligator and EMA
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_confirmed = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Alligator alignment conditions
        # Bullish alignment: JAW > TEETH > LIPS (alligator mouth opening up)
        # Bearish alignment: JAW < TEETH < LIPS (alligator mouth opening down)
        bullish_aligned = curr_jaw > curr_teeth and curr_teeth > curr_lips
        bearish_aligned = curr_jaw < curr_teeth and curr_teeth < curr_lips
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
            # Exit conditions: price below trailing stop OR Alligator loses bullish alignment
            if curr_close < stop_price or not bullish_aligned:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.0 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.0 * curr_atr
            # Exit conditions: price above trailing stop OR Alligator loses bearish alignment
            if curr_close > stop_price or not bearish_aligned:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bullish Alligator alignment AND price > TEETH AND price > 1d EMA50 AND volume confirmed
            if bullish_aligned and curr_close > curr_teeth and curr_close > curr_ema_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_high_since_entry = curr_high
            # Short entry: Bearish Alligator alignment AND price < TEETH AND price < 1d EMA50 AND volume confirmed
            elif bearish_aligned and curr_close < curr_teeth and curr_close < curr_ema_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals