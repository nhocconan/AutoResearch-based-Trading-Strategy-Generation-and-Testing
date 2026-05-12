#!/usr/bin/env python3
name = "1h_4h1d_Trend_Momentum_Filter"
timeframe = "1h"
leverage = 1.0

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
    
    # Multi-timeframe: 4h and 1d for signal direction, 1h for entry timing
    # 4h data
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 1d data
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 4H INDICATORS ===
    # 4h EMA21 for trend
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # 4h RSI14 for momentum
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 4h volume spike
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume_4h > (vol_ma_4h * 1.5)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    # === 1D INDICATORS ===
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d RSI14 for momentum filter
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1H INDICATORS ===
    # 1h ATR14 for volatility filter
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - np.roll(low, 1))
    tr3 = np.abs(np.roll(close, 1) - np.roll(high, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1h RSI14 for entry timing
    delta_1h = np.diff(close, prepend=close[0])
    gain_1h = np.where(delta_1h > 0, delta_1h, 0)
    loss_1h = np.where(delta_1h < 0, -delta_1h, 0)
    avg_gain_1h = pd.Series(gain_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1h = pd.Series(loss_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1h = avg_gain_1h / (avg_loss_1h + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs_1h))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 21, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(atr14[i]) or np.isnan(rsi_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Check session
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 4h uptrend + 4h bullish momentum + 1d uptrend + 1h oversold bounce
            if (close[i] > ema21_4h_aligned[i] and  # 4h price above trend
                rsi_4h_aligned[i] > 50 and         # 4h bullish momentum
                close[i] > ema50_1d_aligned[i] and # 1d price above trend
                rsi_1d_aligned[i] > 50 and         # 1d bullish momentum
                rsi_1h[i] < 30 and                 # 1h oversold (entry timing)
                volume_spike_4h_aligned[i]):       # 4h volume confirmation
                signals[i] = 0.20
                position = 1
            # SHORT: 4h downtrend + 4h bearish momentum + 1d downtrend + 1h overbought bounce
            elif (close[i] < ema21_4h_aligned[i] and  # 4h price below trend
                  rsi_4h_aligned[i] < 50 and         # 4h bearish momentum
                  close[i] < ema50_1d_aligned[i] and # 1d price below trend
                  rsi_1d_aligned[i] < 50 and         # 1d bearish momentum
                  rsi_1h[i] > 70 and                 # 1h overbought (entry timing)
                  volume_spike_4h_aligned[i]):       # 4h volume confirmation
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: 4h trend breaks OR momentum fades
            if (close[i] <= ema21_4h_aligned[i] or  # 4h trend break
                rsi_4h_aligned[i] < 45):            # 4h momentum weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h trend breaks OR momentum fades
            if (close[i] >= ema21_4h_aligned[i] or  # 4h trend break
                rsi_4h_aligned[i] > 55):            # 4h momentum weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals