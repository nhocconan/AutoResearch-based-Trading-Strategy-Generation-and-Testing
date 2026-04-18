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
    
    # Get 4h data for trend filter and volume context
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h EMA34 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate 4h volume ratio for confirmation
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = np.divide(volume_4h, vol_ma_4h, out=np.ones_like(volume_4h), where=vol_ma_4h!=0)
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for regime
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1h RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1h volume spike
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = np.divide(volume, vol_ma_1h, out=np.ones_like(volume), where=vol_ma_1h!=0)
    
    # Calculate 1h ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - low)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_spike[i]) or np.isnan(atr[i]) or
            np.isnan(vol_ratio_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: price above/below 1d EMA50 determines bias
        price_above_ema50 = close[i] > ema50_1d_aligned[i]
        price_below_ema50 = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation: 4h volume ratio > 1.2 AND 1h volume spike > 1.5
        vol_confirm = vol_ratio_4h_aligned[i] > 1.2 and vol_spike[i] > 1.5
        
        # Volatility filter: ATR > 0.5% of price (avoid choppy low-volatility periods)
        vol_filter = atr[i] > 0.005 * close[i]
        
        if position == 0:
            # Long entry: price above 1d EMA50, 4h EMA34 rising, RSI not overbought, volume confirmation
            if (price_above_ema50 and 
                ema34_4h_aligned[i] > ema34_4h_aligned[i-1] and 
                rsi[i] < 60 and 
                vol_confirm and 
                vol_filter):
                signals[i] = 0.20
                position = 1
            # Short entry: price below 1d EMA50, 4h EMA34 falling, RSI not oversold, volume confirmation
            elif (price_below_ema50 and 
                  ema34_4h_aligned[i] < ema34_4h_aligned[i-1] and 
                  rsi[i] > 40 and 
                  vol_confirm and 
                  vol_filter):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below 1d EMA50 or RSI overbought or volatility drops
            if (close[i] < ema50_1d_aligned[i] or 
                rsi[i] > 70 or 
                atr[i] < 0.003 * close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above 1d EMA50 or RSI oversold or volatility drops
            if (close[i] > ema50_1d_aligned[i] or 
                rsi[i] < 30 or 
                atr[i] < 0.003 * close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA50_EMA34_RSI_Volume_Volatility_Filter"
timeframe = "1h"
leverage = 1.0