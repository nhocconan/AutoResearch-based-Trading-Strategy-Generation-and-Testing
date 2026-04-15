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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Calculate 1d EMA20 for trend filter
    ema_20 = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d RSI(14) for momentum filter
    delta = np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = np.abs(np.diff(df_1d['high'].values, prepend=df_1d['high'].values[0]))
    tr2 = np.abs(np.diff(df_1d['low'].values, prepend=df_1d['low'].values[0]))
    tr3 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    prev_close = np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]])
    prev_high = np.concatenate([[df_1d['high'].values[0]], df_1d['high'].values[:-1]])
    prev_low = np.concatenate([[df_1d['low'].values[0]], df_1d['low'].values[:-1]])
    
    camarilla_range = prev_high - prev_low
    camarilla_r4 = prev_close + (camarilla_range * 1.1 / 2)
    camarilla_r3 = prev_close + (camarilla_range * 1.1 / 4)
    camarilla_s3 = prev_close - (camarilla_range * 1.1 / 4)
    camarilla_s4 = prev_close - (camarilla_range * 1.1 / 2)
    
    # Align HTF indicators to 1h timeframe with proper delay
    ema_20_1h = align_htf_to_ltf(prices, df_1d, ema_20)
    rsi_14_1h = align_htf_to_ltf(prices, df_1d, rsi_14)
    atr_14_1h = align_htf_to_ltf(prices, df_1d, atr_14)
    camarilla_r4_1h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_1h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_1h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_1h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_20_1h[i]) or np.isnan(rsi_14_1h[i]) or 
            np.isnan(atr_14_1h[i]) or np.isnan(camarilla_r4_1h[i]) or
            np.isnan(camarilla_r3_1h[i]) or np.isnan(camarilla_s3_1h[i]) or
            np.isnan(camarilla_s4_1h[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 1d trend filter: price above/below daily EMA20
        # 2. 1d momentum filter: RSI not extreme (avoid overbought/oversold)
        # 3. 1h Donchian breakout: price breaks 20-period channel
        # 4. 4h volume confirmation: volume > 1.5x average
        # 5. 1d volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 6. Discrete position sizing: 0.20
        
        # Long conditions: break above Donchian high in uptrend
        if (close[i] > ema_20_1h[i] and          # Daily uptrend filter
            rsi_14_1h[i] < 70 and                # Not overbought
            close[i] > highest_20[i] and         # 1h Donchian breakout
            volume_ratio[i] > 1.5 and            # 4h volume confirmation
            atr_14_1h[i] > 0.005 * close[i]):    # Volatility filter
            signals[i] = 0.20
            
        # Short conditions: break below Donchian low in downtrend
        elif (close[i] < ema_20_1h[i] and        # Daily downtrend filter
              rsi_14_1h[i] > 30 and              # Not oversold
              close[i] < lowest_20[i] and        # 1h Donchian breakdown
              volume_ratio[i] > 1.5 and          # 4h volume confirmation
              atr_14_1h[i] > 0.005 * close[i]):  # Volatility filter
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_Donchian_Breakout_Volume_Trend_Session"
timeframe = "1h"
leverage = 1.0