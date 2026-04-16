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
    
    # === 1h data (HTF for direction) ===
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    volume_1h = df_1h['volume'].values
    
    # 1h Donchian upper and lower bands (20 periods)
    high_20_1h = pd.Series(high_1h).rolling(window=20, min_periods=20).max().values
    low_20_1h = pd.Series(low_1h).rolling(window=20, min_periods=20).min().values
    donchian_upper_1h = align_htf_to_ltf(prices, df_1h, high_20_1h)
    donchian_lower_1h = align_htf_to_ltf(prices, df_1h, low_20_1h)
    
    # 1h EMA20 for trend filter
    close_1h_series = pd.Series(close_1h)
    ema_20_1h = close_1h_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_20_1h)
    
    # === 4h data (HTF for regime) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h ATR for volatility filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # === 15m indicators for entry timing ===
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
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_1h[i]) or np.isnan(donchian_lower_1h[i]) or 
            np.isnan(ema_20_1h_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        upper_1h = donchian_upper_1h[i]
        lower_1h = donchian_lower_1h[i]
        ema_20_1h_val = ema_20_1h_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower OR RSI becomes overbought
            if (price < lower_1h) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper OR RSI becomes oversold
            if (price > upper_1h) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price breaks above Donchian upper AND above EMA20 (trend filter) 
                # AND RSI not overbought AND volume spike AND volatility not too high
                if (price > upper_1h) and (price > ema_20_1h_val) and (rsi_val < 60) and \
                   (vol_ratio_val > 2.0) and (atr_4h_val < np.percentile(atr_4h_aligned[:i+1], 80)):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below Donchian lower AND below EMA20 (trend filter) 
                # AND RSI not oversold AND volume spike AND volatility not too high
                elif (price < lower_1h) and (price < ema_20_1h_val) and (rsi_val > 40) and \
                     (vol_ratio_val > 2.0) and (atr_4h_val < np.percentile(atr_4h_aligned[:i+1], 80)):
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

name = "15m_Donchian_Breakout_EMA20_RSI_Volume"
timeframe = "15m"
leverage = 1.0