#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion using 1d RSI extremes with volume confirmation.
# Long when 1d RSI < 30 (oversold) and price > VWAP, volume > 1.5x average.
# Short when 1d RSI > 70 (overbought) and price < VWAP, volume > 1.5x average.
# Exit when price crosses back to VWAP or RSI returns to neutral zone (40-60).
# Uses 1d for signal direction (avoid whipsaw), 1h for entry timing.
# Target: 20-50 trades/year per symbol to stay within frequency limits.
name = "1h_RSI_VWAP_MeanReversion_Volume"
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
    
    # Get daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 14-period RSI on daily closes
    rsi_period = 14
    delta = np.diff(df_1d['close'].values, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilder_smooth(gain, rsi_period)
    avg_loss = wilder_smooth(loss, rsi_period)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 1h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate VWAP for 1h timeframe
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Get 1h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure RSI and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi = rsi_1d_aligned[i]
        vwap_val = vwap[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: RSI oversold, price above VWAP, volume confirmation
            if rsi < 30 and price > vwap_val and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Short entry: RSI overbought, price below VWAP, volume confirmation
            elif rsi > 70 and price < vwap_val and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below VWAP or RSI returns to neutral
            if price < vwap_val or rsi > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above VWAP or RSI returns to neutral
            if price > vwap_val or rsi < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals