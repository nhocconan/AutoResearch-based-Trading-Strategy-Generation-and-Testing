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
    
    # === 4h data for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate ATR on 4h
    tr_4h = np.maximum(high_4h - low_4h,
                       np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                  np.abs(low_4h - np.roll(close_4h, 1))))
    tr_4h[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Calculate EMA(21) on 4h
    ema_21_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # === 1d data for trend confirmation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA(50) on 1d
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume ratio on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.where(vol_ma_1d > 0, volume_1d / vol_ma_1d, 0)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 1h data for entry timing ===
    # Volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    # RSI(7) for mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=7, min_periods=7).mean().values
    avg_loss = pd.Series(loss).rolling(window=7, min_periods=7).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_7 = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(rsi_7[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_21_4h_val = ema_21_4h_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        vol_ratio_1d_val = vol_ratio_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        rsi_val = rsi_7[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        hour = hours[i]
        
        # Check session: 08-20 UTC
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 4h EMA(21) OR RSI > 70
            if (price < ema_21_4h_val) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 4h EMA(21) OR RSI < 30
            if (price > ema_21_4h_val) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: 1d EMA(50) up + volume confirmation + price near lower BB
            if (ema_50_1d_val > ema_50_1d[max(0, i-16)] if i >= 16 else ema_50_1d_val > ema_50_1d_val) and \
               (vol_ratio_1d_val > 1.5) and (vol_ratio_val > 1.5) and \
               (price <= bb_lower_val * 1.02):  # Allow small tolerance
                signals[i] = 0.20
                position = 1
                continue
            
            # SHORT: 1d EMA(50) down + volume confirmation + price near upper BB
            elif (ema_50_1d_val < ema_50_1d[max(0, i-16)] if i >= 16 else ema_50_1d_val < ema_50_1d_val) and \
                 (vol_ratio_1d_val > 1.5) and (vol_ratio_val > 1.5) and \
                 (price >= bb_upper_val * 0.98):  # Allow small tolerance
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_EMA50_Volume_BB_MeanReversion"
timeframe = "1h"
leverage = 1.0