# 1d: RSI + Weekly Trend + Volume Spike
# Hypothesis: On daily timeframe, use RSI(14) for mean reversion (RSI<30 long, RSI>70 short)
# filtered by weekly trend (price above/below weekly EMA50) and volume spike (>2x 20-day average).
# Works in bull/bear by fading extremes in trending markets. Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly EMA50 once before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: >2x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_ema = ema_50_1w_aligned[i]
        rsi_val = rsi[i]
        vol_conf = vol_spike[i] > 2.0
        
        if position == 0:
            # Long: oversold + above weekly EMA + volume spike
            if rsi_val < 30 and price > weekly_ema and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: overbought + below weekly EMA + volume spike
            elif rsi_val > 70 and price < weekly_ema and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI > 50 or price below weekly EMA
            if rsi_val > 50 or price < weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI < 50 or price above weekly EMA
            if rsi_val < 50 or price > weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0