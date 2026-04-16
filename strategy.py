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
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for regime and support/resistance) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 4h Indicators ===
    # 4h ATR for volatility filter (14 periods)
    tr1_4h = np.abs(high_4h - low_4h)
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr2_4h[0] = np.inf
    tr3_4h[0] = np.inf
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # 4h EMA34 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_34_4h = close_4h_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1d Indicators ===
    # 1d ATR for volatility regime (14 periods)
    tr1_1d = np.abs(high_1d - low_1d)
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = np.inf
    tr3_1d[0] = np.inf
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d RSI for mean reversion signals (14 periods)
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 4h Indicators for entry timing ===
    # RSI(14) on 4h
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    
    # Volume spike detection (4h)
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / (vol_ma_20_4h + 1e-10)
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # Session filter: 08-20 UTC (avoid low liquidity periods)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(rsi_4h[i]) or np.isnan(vol_ratio_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        ema_34_4h_val = ema_34_4h_aligned[i]
        atr_14_4h_val = atr_14_4h_aligned[i]
        atr_14_1d_val = atr_14_1d_aligned[i]
        rsi_1d_val = rsi_1d_aligned[i]
        rsi_4h_val = rsi_4h[i]
        vol_ratio_4h_val = vol_ratio_4h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when RSI becomes overbought OR volatility too high
            if (rsi_4h_val > 70) or (atr_14_4h_val > 1.5 * atr_14_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when RSI becomes oversold OR volatility too high
            if (rsi_4h_val < 30) or (atr_14_4h_val > 1.5 * atr_14_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: RSI 1d oversold (<30) AND 4h RSI recovering from oversold (>40) 
                # AND volume spike AND 4h price above EMA34 (trend alignment) 
                # AND volatility not excessive (4h ATR < 1.5 * 1d ATR)
                if (rsi_1d_val < 30) and (rsi_4h_val > 40) and \
                   (vol_ratio_4h_val > 1.8) and (price > ema_34_4h_val) and \
                   (atr_14_4h_val < 1.5 * atr_14_1d_val):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: RSI 1d overbought (>70) AND 4h RSI declining from overbought (<60) 
                # AND volume spike AND 4h price below EMA34 (trend alignment) 
                # AND volatility not excessive (4h ATR < 1.5 * 1d ATR)
                elif (rsi_1d_val > 70) and (rsi_4h_val < 60) and \
                     (vol_ratio_4h_val > 1.8) and (price < ema_34_4h_val) and \
                     (atr_14_4h_val < 1.5 * atr_14_1d_val):
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

name = "4h_RSI_MeanReversion_Volume_Trend"
timeframe = "4h"
leverage = 1.0