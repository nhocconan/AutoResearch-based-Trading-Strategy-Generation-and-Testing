#!/usr/bin/env python3
# 1h RSI-mean-reversion with 4h trend filter and volume confirmation
# Long when: 1h RSI < 30, 4h EMA50 uptrend, volume > 1.5x 20-bar avg, during 08-20 UTC
# Short when: 1h RSI > 70, 4h EMA50 downtrend, volume > 1.5x 20-bar avg, during 08-20 UTC
# Exit when RSI returns to 50 or volume drops below average
# Position size: 0.20 to manage drawdown and limit trades
# Uses 4h EMA50 for trend filter, 1h RSI for entry timing
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag

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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: 20-period volume MA on 1h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for RSI and EMA
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(rsi_values[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ma_20.iloc[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi_values[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        ema_50 = ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: RSI oversold, 4h uptrend, volume spike
            if rsi_val < 30 and price > ema_50 and vol > 1.5 * vol_ma:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought, 4h downtrend, volume spike
            elif rsi_val > 70 and price < ema_50 and vol > 1.5 * vol_ma:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to 50 or volume drops below average
            if rsi_val >= 50 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI returns to 50 or volume drops below average
            if rsi_val <= 50 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0