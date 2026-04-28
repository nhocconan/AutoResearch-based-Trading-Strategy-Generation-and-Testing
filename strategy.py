# 1h_Volume_Weighted_Momentum_4hTrend
# Hypothesis: Use 4h trend direction (EMA50) as filter, enter on 1h volume-weighted momentum breakouts.
# Volume-weighted RSI filters entries to avoid chop. Designed for 1h to target 60-150 trades over 4 years (15-37/year).
# Works in bull/bear by requiring strong 4h trend and volume confirmation, reducing whipsaws.

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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h VWAP deviation (momentum)
    typical_price = (high + low + close) / 3.0
    vwap_num = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values
    vwap_den = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = np.divide(vwap_num, vwap_den, out=np.zeros_like(vwap_num), where=vwap_den!=0)
    price_vwap_ratio = close / vwap  # >1 = above VWAP (bullish momentum)
    
    # 1h Volume-weighted RSI (avoid chop)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Volume-weighted gains/losses
    vol_gain = gain * volume
    vol_loss = loss * volume
    avg_vol_gain = pd.Series(vol_gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_vol_loss = pd.Series(vol_loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.divide(avg_vol_gain, avg_vol_loss, out=np.zeros_like(avg_vol_gain), where=avg_vol_loss!=0)
    vol_rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data unavailable
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(price_vwap_ratio[i]) or 
            np.isnan(vol_rsi[i])):
            signals[i] = 0.0
            continue
        
        # 4h trend filter: price above/below EMA50
        uptrend_4h = close[i] > ema_50_4h_aligned[i]
        downtrend_4h = close[i] < ema_50_4h_aligned[i]
        
        # Momentum conditions with volume confirmation
        # Long: above VWAP, not overbought, in uptrend
        long_entry = (uptrend_4h and 
                     price_vwap_ratio[i] > 1.002 and  # 0.2% above VWAP
                     vol_rsi[i] < 70 and              # Not overbought
                     volume[i] > np.mean(volume[max(0,i-20):i]) * 1.5 and  # Volume spike
                     in_session[i])
        
        # Short: below VWAP, not oversold, in downtrend
        short_entry = (downtrend_4h and 
                      price_vwap_ratio[i] < 0.998 and  # 0.2% below VWAP
                      vol_rsi[i] > 30 and              # Not oversold
                      volume[i] > np.mean(volume[max(0,i-20):i]) * 1.5 and  # Volume spike
                      in_session[i])
        
        # Exit: opposite VWAP cross or trend change
        long_exit = (not uptrend_4h) or (price_vwap_ratio[i] < 1.0) or (position == 1 and vol_rsi[i] > 70)
        short_exit = (not downtrend_4h) or (price_vwap_ratio[i] > 1.0) or (position == -1 and vol_rsi[i] < 30)
        
        # Handle signals
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Volume_Weighted_Momentum_4hTrend"
timeframe = "1h"
leverage = 1.0