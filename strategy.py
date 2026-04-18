#!/usr/bin/env python3
"""
4h 12-hour Momentum Pullback with Volume Confirmation
Hypothesis: In trending markets, price pulls back to the 12-hour EMA before continuing.
Long when price pulls back to 12h EMA in an uptrend with volume confirmation.
Short when price pulls back to 12h EMA in a downtrend with volume confirmation.
Designed for 20-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA34 on close
    ema_34 = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # 12h trend: EMA34 slope (rising/falling)
    ema_34_prev = np.roll(ema_34, 1)
    ema_34_prev[0] = ema_34[0]
    ema_34_slope = ema_34 - ema_34_prev
    ema_34_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_34_slope)
    
    # Volume spike: 2x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 4h RSI(14) for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_34_slope_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema = ema_34_aligned[i]
        slope = ema_34_slope_aligned[i]
        
        if position == 0:
            # Long: price near 12h EMA in uptrend with volume spike and bullish RSI
            if abs(price - ema) / ema < 0.01 and slope > 0 and volume_spike[i] and rsi_values[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short: price near 12h EMA in downtrend with volume spike and bearish RSI
            elif abs(price - ema) / ema < 0.01 and slope < 0 and volume_spike[i] and rsi_values[i] < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: trend changes or RSI turns bearish
            if slope < 0 or rsi_values[i] < 50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: trend changes or RSI turns bullish
            if slope > 0 or rsi_values[i] > 50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_12hEMA_Pullback_Momentum"
timeframe = "4h"
leverage = 1.0