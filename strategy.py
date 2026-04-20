#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_RSITrend_Confluence_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # === 4h: RSI(14) ===
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss != 0, avg_loss, np.nan)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h[0:14] = np.nan  # Ensure proper warmup
    
    # Align RSI to 1h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # === 1h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume ratio (current vs 24-period average)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma24 > 0, vol_ma24, np.nan)
    
    # ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1h EMA(50) for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        rsi_val = rsi_4h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        ema_val = ema_50[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(vol_ratio_val) or 
            np.isnan(atr_val) or np.isnan(ema_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period median
        atr_median = np.nanmedian(atr[max(0, i-49):i+1]) if i >= 1 else np.nan
        vol_filter = atr_val > atr_median if not np.isnan(atr_median) else False
        
        # RSI trend filter: RSI > 55 for long, RSI < 45 for short
        rsi_filter_long = rsi_val > 55
        rsi_filter_short = rsi_val < 45
        
        if position == 0:
            # Long: RSI bullish with volume confirmation and volatility filter
            if (rsi_filter_long and   # RSI bullish
                vol_ratio_val > 1.8 and    # Volume confirmation
                vol_filter):               # Volatility filter
                signals[i] = 0.20
                position = 1
            # Short: RSI bearish with volume confirmation and volatility filter
            elif (rsi_filter_short and  # RSI bearish
                  vol_ratio_val > 1.8 and    # Volume confirmation
                  vol_filter):               # Volatility filter
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI turns bearish or volatility drops
            if (not rsi_filter_long) or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI turns bullish or volatility drops
            if (not rsi_filter_short) or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals