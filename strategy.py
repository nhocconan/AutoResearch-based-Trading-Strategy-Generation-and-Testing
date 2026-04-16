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
    
    # === Weekly data for trend context ===
    df_week = get_htf_data(prices, '1w')
    close_week = df_week['close'].values
    
    # Weekly EMA(34) for long-term trend
    ema_34_week = pd.Series(close_week).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_week_aligned = align_htf_to_ltf(prices, df_week, ema_34_week)
    
    # === Daily data for entry signals ===
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Daily ATR(14) for volatility
    tr_daily = np.maximum(high_daily - low_daily,
                          np.maximum(np.abs(high_daily - np.roll(close_daily, 1)),
                                     np.abs(low_daily - np.roll(close_daily, 1))))
    tr_daily[0] = high_daily[0] - low_daily[0]
    atr_daily = pd.Series(tr_daily).rolling(window=14, min_periods=14).mean().values
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Daily RSI(14) for momentum filter
    delta = pd.Series(close_daily).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi_values)
    
    # === Daily Donchian Channel (20) for breakout ===
    highest_20 = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_20
    donchian_lower = lowest_20
    
    # === Daily Volume spike filter ===
    vol_ma_20 = pd.Series(volume_daily).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_daily / vol_ma_20
    
    signals = np.zeros(n)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_week_aligned[i]) or np.isnan(atr_daily_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_daily[i]
        ema_trend = ema_34_week_aligned[i]
        atr_val = atr_daily_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower OR RSI overbought
            if (price < donchian_lower[i]) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper OR RSI oversold
            if (price > donchian_upper[i]) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND above weekly EMA34 AND RSI not overbought AND volume spike
            if (price > donchian_upper[i]) and (price > ema_trend) and (rsi_val < 70) and (vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below Donchian lower AND below weekly EMA34 AND RSI not oversold AND volume spike
            elif (price < donchian_lower[i]) and (price < ema_trend) and (rsi_val > 30) and (vol_ratio_val > 1.5):
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

name = "1d_Donchian_WeeklyEMA_RSI_Volume"
timeframe = "1d"
leverage = 1.0