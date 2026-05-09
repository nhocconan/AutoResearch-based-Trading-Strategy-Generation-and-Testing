#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_Direction_RSI_14_Filter_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    close_series = pd.Series(close)
    delta = close_series.diff().abs()
    vol = delta.rolling(window=10, min_periods=10).sum()
    er = delta / vol.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to daily (no need as already daily)
    kama_aligned = kama  # Already on daily timeframe
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Calculate RSI(14)
    delta_rsi = pd.Series(close).diff()
    gain = delta_rsi.where(delta_rsi > 0, 0)
    loss = -delta_rsi.where(delta_rsi < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate volume average for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)  # Need 14 for RSI, 20 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_10_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_aligned[i]
        ema_1w = ema_10_1w_aligned[i]
        rsi_val = rsi[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Price above KAMA, weekly uptrend, RSI not overbought, volume spike
            if close[i] > kama_val and close[i] > ema_1w and rsi_val < 70 and vol > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below KAMA, weekly downtrend, RSI not oversold, volume spike
            elif close[i] < kama_val and close[i] < ema_1w and rsi_val > 30 and vol > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price below KAMA OR weekly trend turns down
            if close[i] < kama_val or close[i] < ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price above KAMA OR weekly trend turns up
            if close[i] > kama_val or close[i] > ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals