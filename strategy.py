#!/usr/bin/env python3
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
    
    # === Daily 144-period EMA for trend direction (long-term trend) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_144 = pd.Series(close_1d).ewm(span=144, adjust=False, min_periods=144).mean().values
    ema_144_aligned = align_htf_to_ltf(prices, df_1d, ema_144)
    
    # === Daily ATR(14) for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # === 12-hour RSI(14) for overbought/oversold signals ===
    # Calculate RSI on 12h closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def rsi_wilder(gain, loss, period):
        gain_avg = np.full_like(gain, np.nan)
        loss_avg = np.full_like(loss, np.nan)
        if len(gain) >= period:
            gain_avg[period-1] = np.mean(gain[:period])
            loss_avg[period-1] = np.mean(loss[:period])
            for i in range(period, len(gain)):
                gain_avg[i] = (gain_avg[i-1] * (period-1) + gain[i]) / period
                loss_avg[i] = (loss_avg[i-1] * (period-1) + loss[i]) / period
        rs = np.where(loss_avg == 0, 0, gain_avg / loss_avg)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_14 = rsi_wilder(gain, loss, 14)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 200
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_144_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(rsi_14[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_trend = ema_144_aligned[i]
        atr_val = atr_14_aligned[i]
        
        # Volatility filter: only trade when ATR > 0.5% of price (avoid choppy low-vol periods)
        vol_filter = atr_val > (price * 0.005)
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Price above long-term EMA + RSI oversold (<30) + volatility filter
            if price > ema_trend and rsi_14[i] < 30 and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price below long-term EMA + RSI overbought (>70) + volatility filter
            elif price < ema_trend and rsi_14[i] > 70 and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal on opposite condition
        elif position == 1:
            # Exit long if price crosses below EMA or RSI becomes overbought
            if price < ema_trend or rsi_14[i] > 70:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price crosses above EMA or RSI becomes oversold
            if price > ema_trend or rsi_14[i] < 30:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA144_RSI14_VolFilter_MeanReversion"
timeframe = "12h"
leverage = 1.0