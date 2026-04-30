#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h EHLERS FISHER TRANSFORM with 12h trend filter and volume confirmation
# Fisher Transform identifies extreme price movements likely to reverse, effective in ranging/weak trending markets
# 12h EMA50 trend filter ensures trades align with intermediate-term direction to avoid counter-trend whipsaws
# Volume spike (1.8x 48-period average) confirms participation on breakouts from extreme levels
# Discrete sizing 0.25 balances profit potential with drawdown control. Target: 80-120 total trades over 4 years (20-30/year).
# Works in bull markets via buying extreme dips in uptrends and selling extreme rallies in uptrends
# Works in bear markets via selling extreme rallies in downtrends and buying extreme dips in downtrends

name = "6h_FisherTransform_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop (MTF Rule #1)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # EHLERS FISHER TRANSFORM on 6h prices
    # Normalize price to [-1, 1] range over 10-period lookback
    def normalize_price(high_arr, low_arr, lookback=10):
        n = len(high_arr)
        nhl = np.zeros(n)  # highest high - lowest low over lookback
        for i in range(lookback-1, n):
            window_high = np.max(high_arr[i-lookback+1:i+1])
            window_low = np.min(low_arr[i-lookback+1:i+1])
            nhl[i] = window_high - window_low
        # For periods before lookback, use available data
        for i in range(lookback-1):
            window_high = np.max(high_arr[0:i+1])
            window_low = np.min(low_arr[0:i+1])
            nhl[i] = window_high - window_low
        # Avoid division by zero
        nhl = np.where(nhl == 0, 1e-10, nhl)
        
        # Calculate normalized price: 2 * ((price - min_low) / (max_high - min_low)) - 1
        norm_price = np.zeros(n)
        for i in range(n):
            window_high = np.max(high_arr[max(0, i-lookback+1):i+1])
            window_low = np.min(low_arr[max(0, i-lookback+1):i+1])
            if window_high != window_low:
                norm_price[i] = 2 * ((close[i] - window_low) / (window_high - window_low)) - 1
            else:
                norm_price[i] = 0
        return np.clip(norm_price, -0.999, 0.999)  # Prevent extreme values
    
    # Fisher Transform formula
    def fisher_transform(norm_price, length=10):
        n = len(norm_price)
        fish = np.zeros(n)
        
        # Value = 0.5 * ln((1 + norm_price) / (1 - norm_price))
        value = np.where(np.abs(norm_price) < 0.999,
                        0.5 * np.log((1 + norm_price) / (1 - norm_price)),
                        0.5 * np.log((1 + 0.999) / (1 - 0.999)) * np.sign(norm_price))
        
        # Smooth value
        smoothed_value = np.zeros(n)
        smoothed_value[0] = value[0]
        for i in range(1, n):
            smoothed_value[i] = 0.5 * value[i] + 0.5 * smoothed_value[i-1]
        
        # Fisher = 0.5 * ln((1 + smoothed_value) / (1 - smoothed_value)) + 0.5 * previous Fisher
        fish[0] = 0.5 * np.log((1 + smoothed_value[0]) / (1 - smoothed_value[0])) if np.abs(smoothed_value[0]) < 0.999 else 0
        for i in range(1, n):
            if np.abs(smoothed_value[i]) < 0.999:
                fish[i] = 0.5 * np.log((1 + smoothed_value[i]) / (1 - smoothed_value[i])) + 0.5 * fish[i-1]
            else:
                fish[i] = (0.5 * np.log((1 + 0.999) / (1 - 0.999)) * np.sign(smoothed_value[i])) + 0.5 * fish[i-1]
        
        return fish
    
    norm_price = normalize_price(high, low, 10)
    fish = fisher_transform(norm_price, 10)
    
    # Volume confirmation: volume > 1.8x 48-period average (48*6h = 288h = 12 days)
    vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    volume_spike = volume > (1.8 * vol_ma_48)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 48)  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(fish[i]) or 
            np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_fish = fish[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike for confirmation
            if curr_volume_spike:
                # Bullish entry: Fisher crosses below -1.5 (extreme oversold) AND price above 12h EMA50 (uptrend)
                if curr_fish <= -1.5 and curr_close > curr_ema_12h:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Fisher crosses above +1.5 (extreme overbought) AND price below 12h EMA50 (downtrend)
                elif curr_fish >= 1.5 and curr_close < curr_ema_12h:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when Fisher crosses above -0.5 (reversal signal) OR price drops below 12h EMA50 (trend change)
            if curr_fish >= -0.5 or curr_close < curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Fisher crosses below +0.5 (reversal signal) OR price rises above 12h EMA50 (trend change)
            if curr_fish <= 0.5 or curr_close > curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals