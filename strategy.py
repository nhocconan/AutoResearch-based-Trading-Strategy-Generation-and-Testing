#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily data (primary) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian channels (20-period)
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper_1d = align_htf_to_ltf(prices, df_1d, high_20_1d)
    donchian_lower_1d = align_htf_to_ltf(prices, df_1d, low_20_1d)
    
    # Daily ATR (14-period) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Weekly data (HTF trend filter)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend direction
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily RSI(14) for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (daily)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_ratio = volume / vol_ma_10
    
    # Time of day filter (UTC 8-20)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Position tracking
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_upper_1d[i]) or np.isnan(donchian_lower_1d[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        upper_1d = donchian_upper_1d[i]
        lower_1d = donchian_lower_1d[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # Exit conditions
        if position == 1:  # Long position
            if price < lower_1d or rsi_val > 70:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            if price > upper_1d or rsi_val < 30:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions (only when flat)
        if position == 0 and in_session:
            # Long: Price breaks above daily Donchian upper, above weekly EMA50,
            # RSI not overbought, volume spike, volatility not extreme
            if (price > upper_1d) and (price > ema_50_1w_val) and (rsi_val < 60) and \
               (vol_ratio_val > 2.0) and (atr_1d_val < np.percentile(atr_1d_aligned[:i+1], 80)):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: Price breaks below daily Donchian lower, below weekly EMA50,
            # RSI not oversold, volume spike, volatility not extreme
            elif (price < lower_1d) and (price < ema_50_1w_val) and (rsi_val > 40) and \
                 (vol_ratio_val > 2.0) and (atr_1d_val < np.percentile(atr_1d_aligned[:i+1], 80)):
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

name = "1d_Donchian_Breakout_EMA50_WeeklyTrend"
timeframe = "1d"
leverage = 1.0