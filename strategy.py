#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_TrendFilter_1hEntry_V1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # HTF: 4h trend filter
    df_4h = get_htf_data(prices, '4h')
    # EMA34 on 4h close
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # HTF: 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h indicators for entry timing
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema34_4h_val = ema34_4h_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        rsi_val = rsi[i]
        vol_filter = volume_filter[i]
        
        # Trend direction: both 4h and 1d EMA aligned
        long_trend = close_val > ema34_4h_val and close_val > ema50_1d_val
        short_trend = close_val < ema34_4h_val and close_val < ema50_1d_val
        
        if position == 0:
            # Long entry: uptrend + RSI pullback + volume
            if long_trend and rsi_val < 40 and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short entry: downtrend + RSI bounce + volume
            elif short_trend and rsi_val > 60 and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend breaks or RSI overbought
            if not long_trend or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend breaks or RSI oversold
            if not short_trend or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals