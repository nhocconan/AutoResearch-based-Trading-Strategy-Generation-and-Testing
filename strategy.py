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
    
    # === 1d data (primary) ===
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA50 for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # RSI(14) for momentum
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
    
    # === 1w data (HTF for trend direction) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w EMA20 for long-term trend
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === 1d ATR for volatility filter ===
    high_1d = high
    low_1d = low
    close_1d = close
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(atr_1d[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        ema_50_val = ema_50[i]
        ema_20_1w_val = ema_20_1w_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        atr_1d_val = atr_1d[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower OR RSI overbought
            if (price < lower) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper OR RSI oversold
            if (price > upper) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price breaks above Donchian upper AND above EMA50 (trend filter) 
                # AND weekly EMA20 confirms uptrend AND RSI not overbought AND volume spike
                if (price > upper) and (price > ema_50_val) and (ema_20_1w_val > ema_20_1w_aligned[i-1]) and \
                   (rsi_val < 60) and (vol_ratio_val > 2.0):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below Donchian lower AND below EMA50 (trend filter) 
                # AND weekly EMA20 confirms downtrend AND RSI not oversold AND volume spike
                elif (price < lower) and (price < ema_50_val) and (ema_20_1w_val < ema_20_1w_aligned[i-1]) and \
                     (rsi_val > 40) and (vol_ratio_val > 2.0):
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

name = "1d_Donchian_Breakout_EMA50_RSI_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0