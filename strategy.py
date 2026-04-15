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
    
    # Get daily HTF data once before loop (4h primary, 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate 4h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    prev_close = np.concatenate([[daily_close[0]], daily_close[:-1]])
    prev_high = np.concatenate([[daily_high[0]], daily_high[:-1]])
    prev_low = np.concatenate([[daily_low[0]], daily_low[:-1]])
    
    camarilla_range = prev_high - prev_low
    camarilla_r4 = prev_close + (camarilla_range * 1.1 / 2)
    camarilla_r3 = prev_close + (camarilla_range * 1.1 / 4)
    camarilla_s3 = prev_close - (camarilla_range * 1.1 / 4)
    camarilla_s4 = prev_close - (camarilla_range * 1.1 / 2)
    
    # Calculate 1d EMA20 for trend filter
    ema_20 = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d RSI(14) for momentum filter
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to 4h timeframe with proper delay
    ema_20_4h = align_htf_to_ltf(prices, df_1d, ema_20)
    rsi_14_4h = align_htf_to_ltf(prices, df_1d, rsi_14)
    camarilla_r4_4h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_4h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h[i]) or np.isnan(rsi_14_4h[i]) or 
            np.isnan(camarilla_r4_4h[i]) or np.isnan(camarilla_r3_4h[i]) or
            np.isnan(camarilla_s3_4h[i]) or np.isnan(camarilla_s4_4h[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i]) or
            np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 1d trend filter: price above/below daily EMA20
        # 2. 1d momentum filter: RSI not extreme (avoid overbought/oversold)
        # 3. 4h Donchian breakout: price breaks above/below 20-period channel
        # 4. 4h volume confirmation: volume > 1.5x average
        # 5. 1d volatility filter: ATR > 0 (always true, but keeps structure)
        # 6. Discrete position sizing: 0.25
        
        # Long conditions: break above Donchian high in uptrend
        if (close[i] > ema_20_4h[i] and          # Daily uptrend filter
            rsi_14_4h[i] < 70 and                # Not overbought
            close[i] > highest_20[i] and         # 4h Donchian breakout
            volume_ratio[i] > 1.5):              # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: break below Donchian low in downtrend
        elif (close[i] < ema_20_4h[i] and        # Daily downtrend filter
              rsi_14_4h[i] > 30 and              # Not oversold
              close[i] < lowest_20[i] and        # 4h Donchian breakdown
              volume_ratio[i] > 1.5):            # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0