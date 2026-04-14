#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    if len(weekly_close) >= 50:
        weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False).mean().values
    else:
        weekly_ema50 = np.full(len(weekly_close), np.nan)
    
    # Daily Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    donch_mid = (donch_high + donch_low) / 2
    
    # Daily volume filter: 20-period average
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count >= 20:
            vol_ma[i] = vol_sum / vol_count
            vol_sum -= volume[i-19]
            vol_count -= 1
    
    # Daily RSI(14) for overbought/oversold
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    gain_sum = 0
    loss_sum = 0
    for i in range(n):
        gain_sum += gain[i]
        loss_sum += loss[i]
        if i >= 13:
            if i == 13:
                avg_gain[i] = gain_sum / 14
                avg_loss[i] = loss_sum / 14
            else:
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
            if avg_loss[i] != 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100
            # RSI not stored in array for simplicity, calculated on fly
    
    # Align weekly trend
    weekly_trend = weekly_close > weekly_ema50
    weekly_trend_arr = weekly_trend.astype(float)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_arr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(19, n):
        # Skip if any indicator not ready
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or np.isnan(vol_ma[i]):
            continue
        
        weekly_trend_val = weekly_trend_aligned[i]
        if np.isnan(weekly_trend_val):
            continue
        
        # Calculate RSI for current bar
        if i >= 14:
            # Recalculate RSI components up to i
            gain_sum_rsi = np.sum(gain[max(0, i-13):i+1])
            loss_sum_rsi = np.sum(loss[max(0, i-13):i+1])
            if loss_sum_rsi == 0:
                rsi_val = 100
            else:
                rs_val = gain_sum_rsi / loss_sum_rsi
                rsi_val = 100 - (100 / (1 + rs_val))
        else:
            continue
        
        if position == 0:
            # Long: Break above Donchian high, volume spike, weekly uptrend, RSI not overbought
            if close[i] > donch_high[i] and volume[i] > vol_ma[i] * 1.5 and weekly_trend_val > 0.5 and rsi_val < 70:
                position = 1
                signals[i] = position_size
            # Short: Break below Donchian low, volume spike, weekly downtrend, RSI not oversold
            elif close[i] < donch_low[i] and volume[i] > vol_ma[i] * 1.5 and weekly_trend_val < 0.5 and rsi_val > 30:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price closes below Donchian middle OR RSI overbought
            if close[i] < donch_mid[i] and close[i-1] >= donch_mid[i-1]:
                position = 0
                signals[i] = 0.0
            elif rsi_val > 70:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price closes above Donchian middle OR RSI oversold
            if close[i] > donch_mid[i] and close[i-1] <= donch_mid[i-1]:
                position = 0
                signals[i] = 0.0
            elif rsi_val < 30:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian_Breakout_Volume_WeeklyTrend_RSI"
timeframe = "1d"
leverage = 1.0