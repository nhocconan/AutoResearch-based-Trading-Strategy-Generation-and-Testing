#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy with 4h trend filter and 1d regime filter.
# Long when: 1h RSI > 55, 4h close > 4h EMA50, 1d ADX < 25 (low volatility regime)
# Short when: 1h RSI < 45, 4h close < 4h EMA50, 1d ADX < 25
# Uses session filter (08-20 UTC) to avoid low-volume periods.
# Discrete position size 0.20 to minimize churn. Target: 20-40 trades/year per symbol.
name = "1h_RSI4hEMA50_1dADX_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute hour filter for session (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros_like(close)
        valid = (plus_di[period:] + minus_di[period:]) > 0
        dx[period:] = np.where(valid, 100 * np.abs(plus_di[period:] - minus_di[period:]) / (plus_di[period:] + minus_di[period:]), 0)
        
        adx = np.zeros_like(close)
        if len(dx) >= 2*period+1:
            adx[2*period] = np.mean(dx[period:2*period+1])
            for i in range(2*period+1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate RSI(14) on 1h
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50_4h_val = ema_50_4h_aligned[i]
        adx_1d_val = adx_1d_aligned[i]
        rsi_val = rsi[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.3 * vol_ma
        
        # Regime filter: low volatility (ADX < 25) for mean reversion
        low_volatility = adx_1d_val < 25
        
        if position == 0:
            # Enter long if RSI > 55, 4h uptrend, low volatility, volume confirmation
            if (rsi_val > 55 and price > ema_50_4h_val and 
                low_volatility and volume_confirmed):
                signals[i] = 0.20
                position = 1
            # Enter short if RSI < 45, 4h downtrend, low volatility, volume confirmation
            elif (rsi_val < 45 and price < ema_50_4h_val and 
                  low_volatility and volume_confirmed):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long when RSI < 50 or trend breaks
            if rsi_val < 50 or price < ema_50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short when RSI > 50 or trend breaks
            if rsi_val > 50 or price > ema_50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals