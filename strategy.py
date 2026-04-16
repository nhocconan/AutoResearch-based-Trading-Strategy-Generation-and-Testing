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
    
    # === Weekly data for trend filter (1w) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian channels (20 periods) for trend direction
    high_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_upper_1w = align_htf_to_ltf(prices, df_1w, high_20_1w)
    donchian_lower_1w = align_htf_to_ltf(prices, df_1w, low_20_1w)
    
    # === Daily data for entry triggers ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Daily ATR for volatility filter
    high_1d_arr = df_1d['high'].values
    low_1d_arr = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = np.abs(high_1d_arr - low_1d_arr)
    tr2 = np.abs(high_1d_arr - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d_arr - np.roll(close_1d_arr, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily 50-period EMA for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily RSI(14) for momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Daily volume spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_1w[i]) or np.isnan(donchian_lower_1w[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_1w = donchian_upper_1w[i]
        lower_1w = donchian_lower_1w[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below weekly Donchian lower OR RSI becomes overbought
            if (price < lower_1w) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above weekly Donchian upper OR RSI becomes oversold
            if (price > upper_1w) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above weekly Donchian upper AND above daily EMA50 (trend filter) 
            # AND RSI not overbought AND volume spike AND volatility not extreme
            if (price > upper_1w) and (price > ema_50_1d_val) and (rsi_val < 65) and \
               (vol_ratio_val > 1.8) and (atr_1d_val < np.percentile(atr_1d_aligned[:i+1], 85)):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below weekly Donchian lower AND below daily EMA50 (trend filter) 
            # AND RSI not oversold AND volume spike AND volatility not extreme
            elif (price < lower_1w) and (price < ema_50_1d_val) and (rsi_val > 35) and \
                 (vol_ratio_val > 1.8) and (atr_1d_val < np.percentile(atr_1d_aligned[:i+1], 85)):
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

name = "1d_WeeklyDonchian_EMA50_RSI_Volume"
timeframe = "1d"
leverage = 1.0