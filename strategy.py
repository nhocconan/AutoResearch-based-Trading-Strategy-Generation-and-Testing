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
    volume = prices['volume'].values
    
    # === 1h data for momentum and structure ===
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    volume_1h = df_1h['volume'].values
    
    # 1h EMA14 for momentum (fast)
    close_1h_series = pd.Series(close_1h)
    ema_14_1h = close_1h_series.ewm(span=14, min_periods=14, adjust=False).mean().values
    ema_14_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_14_1h)
    
    # 1h EMA50 for trend (slow)
    ema_50_1h = close_1h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    # 1h ATR(14) for volatility filter
    tr1 = np.abs(high_1h - low_1h)
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1h_aligned = align_htf_to_ltf(prices, df_1h, atr_1h)
    
    # === 1d data for regime and pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d ATR(14) for volatility regime filter
    tr1d = np.abs(high_1d - low_1d)
    tr2d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2d[0] = np.inf
    tr3d[0] = np.inf
    tr_d = np.maximum(tr1d, np.maximum(tr2d, tr3d))
    atr_1d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d daily range for volatility context
    daily_range = (high_1d - low_1d)
    avg_daily_range = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    avg_daily_range_aligned = align_htf_to_ltf(prices, df_1d, avg_daily_range)
    
    # === 6h indicators for entry timing ===
    # RSI(14) on 6h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # Session filter: 08-20 UTC (most liquid session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_14_1h_aligned[i]) or np.isnan(ema_50_1h_aligned[i]) or 
            np.isnan(atr_1h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(avg_daily_range_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        ema_14_1h_val = ema_14_1h_aligned[i]
        ema_50_1h_val = ema_50_1h_aligned[i]
        atr_1h_val = atr_1h_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        avg_daily_range_val = avg_daily_range_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when momentum weakens OR RSI overbought OR volatility too high
            if (price < ema_14_1h_val) or (rsi_val > 70) or (atr_1h_val > 1.5 * atr_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when momentum strengthens against short OR RSI oversold OR volatility too high
            if (price > ema_14_1h_val) or (rsi_val < 30) or (atr_1h_val > 1.5 * atr_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # Volatility regime filter: avoid extremely low volatility (chop)
                vol_regime_ok = atr_1h_val > 0.3 * avg_daily_range_val
                
                # LONG: Fast EMA above slow EMA (bullish momentum) AND 
                # RSI not overbought AND volume spike AND volatility regime OK
                if (ema_14_1h_val > ema_50_1h_val) and (rsi_val < 60) and \
                   (vol_ratio_val > 2.0) and vol_regime_ok:
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Fast EMA below slow EMA (bearish momentum) AND 
                # RSI not oversold AND volume spike AND volatility regime OK
                elif (ema_14_1h_val < ema_50_1h_val) and (rsi_val > 40) and \
                     (vol_ratio_val > 2.0) and vol_regime_ok:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_EMA14_50_Momentum_Volume"
timeframe = "6h"
leverage = 1.0