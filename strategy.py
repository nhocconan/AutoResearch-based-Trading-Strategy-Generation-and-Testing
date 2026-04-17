#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Primary timeframe: 1d (from price data)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Multi-timeframe: load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need enough data for weekly indicators
        return np.zeros(n)
    
    # === Weekly Trend Filter: 20-period EMA ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === Weekly Momentum: RSI(14) ===
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    if len(gain) > 0:
        avg_gain[0] = gain[0]
        avg_loss[0] = loss[0]
        for i in range(1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # === Daily Price Action: Price vs Weekly EMA ===
    # Long when price above weekly EMA, short when below
    price_above_weekly_ema = close > ema_20_1w_aligned
    price_below_weekly_ema = close < ema_20_1w_aligned
    
    # === Daily Momentum Filter: RSI(14) on daily data ===
    delta_d = np.diff(close, prepend=close[0])
    gain_d = np.where(delta_d > 0, delta_d, 0)
    loss_d = np.where(delta_d < 0, -delta_d, 0)
    
    avg_gain_d = np.zeros_like(gain_d)
    avg_loss_d = np.zeros_like(loss_d)
    if len(gain_d) > 0:
        avg_gain_d[0] = gain_d[0]
        avg_loss_d[0] = loss_d[0]
        for i in range(1, len(gain_d)):
            avg_gain_d[i] = (avg_gain_d[i-1] * 13 + gain_d[i]) / 14
            avg_loss_d[i] = (avg_loss_d[i-1] * 13 + loss_d[i]) / 14
    
    rs_d = np.divide(avg_gain_d, avg_loss_d, out=np.zeros_like(avg_gain_d), where=avg_loss_d!=0)
    rsi_daily = 100 - (100 / (1 + rs_d))
    
    # === Volume Confirmation: 20-period average ===
    vol_ma_20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1]) if i > 0 else volume[0]
    
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position to avoid whipsaw
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(rsi_daily[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long conditions:
            # 1. Price above weekly EMA (uptrend)
            # 2. Weekly RSI not overbought (< 60)
            # 3. Daily RSI not extreme (< 50 to avoid buying strength)
            # 4. Volume confirmation
            if (price_above_weekly_ema[i] and 
                rsi_1w_aligned[i] < 60 and 
                rsi_daily[i] < 50 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short conditions:
            # 1. Price below weekly EMA (downtrend)
            # 2. Weekly RSI not oversold (> 40)
            # 3. Daily RSI not extreme (> 50 to avoid selling weakness)
            # 4. Volume confirmation
            elif (price_below_weekly_ema[i] and 
                  rsi_1w_aligned[i] > 40 and 
                  rsi_daily[i] > 50 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or loss of momentum
        elif position == 1:
            # Exit long: price crosses below weekly EMA OR weekly RSI overbought
            if (close[i] < ema_20_1w_aligned[i] or rsi_1w_aligned[i] >= 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly EMA OR weekly RSI oversold
            if (close[i] > ema_20_1w_aligned[i] or rsi_1w_aligned[i] <= 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyTrend_Momentum_Volume"
timeframe = "1d"
leverage = 1.0