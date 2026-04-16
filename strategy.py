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
    
    # === 4h data for signal direction ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA34 for trend
    close_4h_series = pd.Series(close_4h)
    ema_34_4h = close_4h_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 4h Donchian(20) for structure
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_4h = align_htf_to_ltf(prices, df_4h, high_20_4h)
    donchian_lower_4h = align_htf_to_ltf(prices, df_4h, low_20_4h)
    
    # === 1d data for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d ATR for volatility regime
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1h indicators for entry timing ===
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (1h)
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
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(donchian_upper_4h[i]) or 
            np.isnan(donchian_lower_4h[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_34_4h_val = ema_34_4h_aligned[i]
        upper_4h = donchian_upper_4h[i]
        lower_4h = donchian_lower_4h[i]
        atr_1d_val = atr_1d_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        hour = hours[i]
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hour <= 20):
            if position == 0:
                signals[i] = 0.0
            else:
                signals[i] = 0.2 if position == 1 else -0.2
            continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 4h EMA34 OR RSI overbought
            if (price < ema_34_4h_val) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 4h EMA34 OR RSI oversold
            if (price > ema_34_4h_val) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade in low volatility regime (ATR below median)
            if atr_1d_val < np.percentile(atr_1d_aligned[:i+1], 50):
                # LONG: Price above 4h EMA34 AND breaks above 4h Donchian upper 
                # AND RSI not overbought AND volume spike
                if (price > ema_34_4h_val) and (price > upper_4h) and (rsi_val < 60) and \
                   (vol_ratio_val > 2.0):
                    signals[i] = 0.2
                    position = 1
                    continue
                
                # SHORT: Price below 4h EMA34 AND breaks below 4h Donchian lower 
                # AND RSI not oversold AND volume spike
                elif (price < ema_34_4h_val) and (price < lower_4h) and (rsi_val > 40) and \
                     (vol_ratio_val > 2.0):
                    signals[i] = -0.2
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.2
        elif position == -1:
            signals[i] = -0.2
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_EMA34_4hTrend_Donchian20_Volume_Session"
timeframe = "1h"
leverage = 1.0