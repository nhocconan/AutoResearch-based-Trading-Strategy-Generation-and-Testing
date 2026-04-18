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
    
    # Daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 4-hour data for trend context
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate ATR (4h) for volatility normalization
    def calculate_atr(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period] = np.nanmean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_4h = calculate_atr(df_4h['high'].values, df_4h['low'].values, close_4h, 14)
    
    # Calculate 4h EMA(34) for trend filter
    if len(close_4h) >= 34:
        ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False).mean().values
    else:
        ema_34_4h = np.full_like(close_4h, np.nan)
    
    # Align 4h data to 1h timeframe
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Daily ATR for volatility-based position sizing
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily EMA(50) for trend filter
    if len(close_1d) >= 50:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    else:
        ema_50_1d = np.full_like(close_1d, np.nan)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily RSI(14) for overbought/oversold conditions
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        if len(close) >= period:
            avg_gain[period] = np.nanmean(gain[1:period+1])
            avg_loss[period] = np.nanmean(loss[1:period+1])
            for i in range(period+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.full_like(close, np.nan)
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility-adjusted position size (0.25 base)
        vol_factor = atr_1d_aligned[i] / np.nanmedian(atr_1d_aligned[max(0, i-50):i+1])
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        base_size = 0.25
        position_size = base_size * vol_factor
        position_size = min(position_size, 0.35)  # Cap at 0.35
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filters: price above 4h EMA(34) AND above daily EMA(50)
        uptrend = close[i] > ema_34_4h_aligned[i] and close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_34_4h_aligned[i] and close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: RSI oversold (<30) + uptrend + volume
            if rsi_1d_aligned[i] < 30 and uptrend and vol_confirm:
                signals[i] = position_size
                position = 1
            # Short: RSI overbought (>70) + downtrend + volume
            elif rsi_1d_aligned[i] > 70 and downtrend and vol_confirm:
                signals[i] = -position_size
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought OR trend reversal OR volatility spike
            if (rsi_1d_aligned[i] > 70 or 
                close[i] < ema_34_4h_aligned[i] - 0.5 * atr_4h_aligned[i] or
                atr_4h_aligned[i] > 2.0 * np.nanmedian(atr_4h_aligned[max(0, i-20):i+1])):
                signals[i] = -position_size  # reverse to short
                position = -1
            else:
                signals[i] = position_size
        
        elif position == -1:
            # Short exit: RSI oversold OR trend reversal OR volatility spike
            if (rsi_1d_aligned[i] < 30 or 
                close[i] > ema_34_4h_aligned[i] + 0.5 * atr_4h_aligned[i] or
                atr_4h_aligned[i] > 2.0 * np.nanmedian(atr_4h_aligned[max(0, i-20):i+1])):
                signals[i] = position_size  # reverse to long
                position = 1
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_RSI_Trend_Volume_Reversal"
timeframe = "1h"
leverage = 1.0