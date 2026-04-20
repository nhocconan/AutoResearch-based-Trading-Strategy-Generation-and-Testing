#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly EMA filter and volume confirmation
# Works in bull: catches breakouts above weekly EMA (uptrend)
# Works in bear: catches breakdowns below weekly EMA (downtrend)
# Target: 20-50 trades/year on 1d timeframe
# Uses discrete position sizing (0.25) to minimize churn

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Donchian channels and ATR
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Daily Donchian channels (20-period)
    # Upper band: highest high over last 20 days
    # Lower band: lowest low over last 20 days
    high_20 = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # Daily ATR (14) for volatility filter
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to minute timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_daily, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_daily, low_20)
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    # Weekly EMA(20) for trend filter
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Main timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(atr_daily_aligned[i]) or np.isnan(ema20_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper_band = high_20_aligned[i]
        lower_band = low_20_aligned[i]
        atr = atr_daily_aligned[i]
        weekly_ema = ema20_weekly_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma_recent = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma_recent
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr > 0.001 * price  # Avoid when ATR < 0.1% of price
        
        if position == 0:
            # Long: price breaks above upper Donchian band, above weekly EMA, with volume
            if (price > upper_band and price > weekly_ema and vol_ok and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band, below weekly EMA, with volume
            elif (price < lower_band and price < weekly_ema and vol_ok and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian band OR closes below weekly EMA
            if price < lower_band or price < weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian band OR closes above weekly EMA
            if price > upper_band or price > weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0