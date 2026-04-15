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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    prev_close = np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]].values)
    prev_high = np.concatenate([[df_12h['high'].iloc[0]], df_12h['high'].iloc[:-1]].values)
    prev_low = np.concatenate([[df_12h['low'].iloc[0]], df_12h['low'].iloc[:-1]].values)
    
    camarilla_range = prev_high - prev_low
    camarilla_r4 = prev_close + (camarilla_range * 1.1 / 2)
    camarilla_r3 = prev_close + (camarilla_range * 1.1 / 4)
    camarilla_s3 = prev_close - (camarilla_range * 1.1 / 4)
    camarilla_s4 = prev_close - (camarilla_range * 1.1 / 2)
    
    # Calculate 12h EMA20 for trend filter
    ema_20 = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 12h RSI(14) for momentum filter
    delta = np.diff(df_12h['close'].values, prepend=df_12h['close'].iloc[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to 4h timeframe with proper delay
    ema_20_4h = align_htf_to_ltf(prices, df_12h, ema_20)
    rsi_14_4h = align_htf_to_ltf(prices, df_12h, rsi_14)
    camarilla_r4_4h = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_r3_4h = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_s4_4h = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h[i]) or np.isnan(rsi_14_4h[i]) or 
            np.isnan(camarilla_r4_4h[i]) or np.isnan(camarilla_r3_4h[i]) or
            np.isnan(camarilla_s3_4h[i]) or np.isnan(camarilla_s4_4h[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 12h trend filter: price above/below 12h EMA20
        # 2. 12h momentum filter: RSI not extreme (avoid overbought/oversold)
        # 3. 4h Camarilla breakout: price breaks R4/S4 for continuation
        # 4. 4h volume confirmation: volume > 1.5x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: break above R4 in uptrend
        if (close[i] > ema_20_4h[i] and          # 12h uptrend filter
            rsi_14_4h[i] < 70 and                # Not overbought
            close[i] > camarilla_r4_4h[i] and    # Camarilla R4 breakout
            volume_ratio[i] > 1.5):              # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: break below S4 in downtrend
        elif (close[i] < ema_20_4h[i] and        # 12h downtrend filter
              rsi_14_4h[i] > 30 and              # Not oversold
              close[i] < camarilla_s4_4h[i] and  # Camarilla S4 breakdown
              volume_ratio[i] > 1.5):            # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_12h_Camarilla_R3_S3_R4_S4_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0