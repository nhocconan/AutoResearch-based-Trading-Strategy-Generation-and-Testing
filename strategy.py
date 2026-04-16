#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (HTF for direction) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4x ATR for stop calculation
    high_low = high_4h - low_4h
    high_prev_close = np.abs(high_4h - np.roll(close_4h, 1))
    low_prev_close = np.abs(low_4h - np.roll(close_4h, 1))
    high_prev_close[0] = np.inf
    low_prev_close[0] = np.inf
    tr = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # 4h EMA20 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_20_4h = close_4h_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # === 1d data (HTF for regime) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Bollinger Bands for volatility regime
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # === 4h indicators for entry timing ===
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_ratio = volume / vol_ma_10
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(bb_width_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        ema_20_4h_val = ema_20_4h_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        bb_width_1d_val = bb_width_1d_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below EMA20 OR RSI becomes overbought OR volatility too high
            if (price < ema_20_4h_val) or (rsi_val > 70) or (bb_width_1d_val > np.percentile(bb_width_1d_aligned[:i+1], 80)):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above EMA20 OR RSI becomes oversold OR volatility too high
            if (price > ema_20_4h_val) or (rsi_val < 30) or (bb_width_1d_val > np.percentile(bb_width_1d_aligned[:i+1], 80)):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price above EMA20 AND RSI not overbought AND volume spike AND volatility not too high
                if (price > ema_20_4h_val) and (rsi_val < 60) and \
                   (vol_ratio_val > 2.0) and (bb_width_1d_val < np.percentile(bb_width_1d_aligned[:i+1], 80)):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price below EMA20 AND RSI not oversold AND volume spike AND volatility not too high
                elif (price < ema_20_4h_val) and (rsi_val > 40) and \
                     (vol_ratio_val > 2.0) and (bb_width_1d_val < np.percentile(bb_width_1d_aligned[:i+1], 80)):
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_EMA20_RSI_Volume_BBWidth"
timeframe = "4h"
leverage = 1.0