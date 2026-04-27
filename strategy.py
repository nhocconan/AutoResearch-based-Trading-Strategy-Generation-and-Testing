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
    
    # Get 1d data for daily ATR and moving averages
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = np.full(len(tr1), np.nan)
    for i in range(14, len(tr)):
        atr_14[i-1] = np.mean(tr[i-14:i])
    atr_14 = np.concatenate([[np.nan]*14, atr_14])
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4-period RSI for momentum
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(4, n):
        if i == 4:
            avg_gain[i] = np.mean(gain[0:4])
            avg_loss[i] = np.mean(loss[0:4])
        else:
            avg_gain[i] = (avg_gain[i-1] * 3 + gain[i-1]) / 4
            avg_loss[i] = (avg_loss[i-1] * 3 + loss[i-1]) / 4
    rs = np.divide(avg_gain, avg_los, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h volume MA(20)
    vol_12h = df_12h['volume'].values
    vol_ma_20 = np.full(len(vol_12h), np.nan)
    for i in range(20, len(vol_12h)):
        vol_ma_20[i] = np.mean(vol_12h[i-20:i])
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        # Trend filter: price above/below daily EMA50
        uptrend = price > ema_50_1d_aligned[i]
        downtrend = price < ema_50_1d_aligned[i]
        
        # Volatility filter: current volatility below 1.5x ATR (avoid choppy markets)
        low_volatility = atr_14_aligned[i] < (np.nanmedian(atr_14_aligned[max(0, i-50):i]) * 1.5) if not np.isnan(np.nanmedian(atr_14_aligned[max(0, i-50):i])) else True
        
        # Momentum filter: RSI in neutral range (40-60) to avoid extremes
        momentum_ok = (rsi[i] >= 40) and (rsi[i] <= 60)
        
        # Volume confirmation: moderate volume (0.8x to 2.5x average)
        volume_ok = (vol_ratio >= 0.8) and (vol_ratio <= 2.5)
        
        if position == 0:
            # Long entry: uptrend + low volatility + momentum OK + volume OK
            if uptrend and low_volatility and momentum_ok and volume_ok:
                signals[i] = size
                position = 1
            # Short entry: downtrend + low volatility + momentum OK + volume OK
            elif downtrend and low_volatility and momentum_ok and volume_ok:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: trend reversal or volatility spike
            if not uptrend or not low_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: trend reversal or volatility spike
            if not downtrend or not low_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_DailyTrend_Volatility_Momentum_Volume"
timeframe = "4h"
leverage = 1.0