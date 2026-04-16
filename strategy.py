#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # 4-hour Donchian channels (20 periods)
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_4h = align_htf_to_ltf(prices, df_4h, high_20_4h)
    donchian_lower_4h = align_htf_to_ltf(prices, df_4h, low_20_4h)
    
    # 4-hour EMA for trend filter (10 periods)
    close_4h_series = pd.Series(close_4h)
    ema_10_4h = close_4h_series.ewm(span=10, min_periods=10, adjust=False).mean().values
    ema_10_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_10_4h)
    
    # === 12h data (HTF for regime and confirmation) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12-hour ATR for volatility filter (14 periods)
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # 12-hour EMA for trend filter (20 periods)
    close_12h_series = pd.Series(close_12h)
    ema_20_12h = close_12h_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # === 4h indicators for entry timing ===
    # RSI(14) on 4h close
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Volume spike detection on 4h
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / (vol_ma_20_4h + 1e-10)
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_4h[i]) or np.isnan(donchian_lower_4h[i]) or 
            np.isnan(ema_10_4h_aligned[i]) or np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(vol_ratio_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        upper_4h = donchian_upper_4h[i]
        lower_4h = donchian_lower_4h[i]
        ema_10_4h_val = ema_10_4h_aligned[i]
        ema_20_12h_val = ema_20_12h_aligned[i]
        atr_12h_val = atr_12h_aligned[i]
        rsi_val = rsi_4h_aligned[i]
        vol_ratio_val = vol_ratio_4h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 4h EMA10 OR RSI becomes overbought
            if (price < ema_10_4h_val) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 4h EMA10 OR RSI becomes oversold
            if (price > ema_10_4h_val) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price breaks above 4h Donchian upper AND above 4h EMA10
                # AND 12h EMA20 confirms uptrend AND RSI not overbought AND volume spike
                # AND volatility not too high (below 80th percentile)
                if (price > upper_4h) and (price > ema_10_4h_val) and (price > ema_20_12h_val) and \
                   (rsi_val < 60) and (vol_ratio_val > 1.8) and \
                   (atr_12h_val < np.percentile(atr_12h_aligned[:i+1], 80)):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below 4h Donchian lower AND below 4h EMA10
                # AND 12h EMA20 confirms downtrend AND RSI not oversold AND volume spike
                # AND volatility not too high
                elif (price < lower_4h) and (price < ema_10_4h_val) and (price < ema_20_12h_val) and \
                     (rsi_val > 40) and (vol_ratio_val > 1.8) and \
                     (atr_12h_val < np.percentile(atr_12h_aligned[:i+1], 80)):
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

name = "4h_Donchian_EMA10_12hEMA20_RSI_Volume"
timeframe = "4h"
leverage = 1.0