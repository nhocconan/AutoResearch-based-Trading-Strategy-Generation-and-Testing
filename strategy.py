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
    
    # === 1d Close for indicators ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d RSI (14) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    period = 14
    for i in range(len(gain)):
        if i < period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[avg_loss == 0] = 100
    
    # === 1d ATR (14) ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < period:
            if i == 0:
                atr[i] = tr[i]
            else:
                atr[i] = (atr[i-1] * (i-1) + tr[i]) / i
        else:
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    # === 1d Bollinger Bands (20,2) ===
    sma_20 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 19:
            sma_20[i] = np.mean(close_1d[i-19:i+1])
        elif i > 0:
            sma_20[i] = np.mean(close_1d[max(0, i-9):i+1])
        else:
            sma_20[i] = close_1d[0]
    
    std_20 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 19:
            std_20[i] = np.std(close_1d[i-19:i+1])
        elif i > 0:
            std_20[i] = np.std(close_1d[max(0, i-9):i+1])
        else:
            std_20[i] = 0.0
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # === 1d BB Width percentile (20-period) for regime detection ===
    bb_width = (upper_bb - lower_bb) / sma_20
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(len(bb_width)):
        if i >= 19:
            window = bb_width[i-19:i+1]
            rank = np.sum(window <= bb_width[i]) / len(window)
            bb_width_percentile[i] = rank * 100
        elif i > 0:
            window = bb_width[max(0, i-9):i+1]
            rank = np.sum(window <= bb_width[i]) / len(window)
            bb_width_percentile[i] = rank * 100
        else:
            bb_width_percentile[i] = 50.0
    
    # === 1d Volume spike (volume > 1.5x 20-period average) ===
    vol_ma_20 = np.full_like(volume_1d, np.nan)
    for i in range(len(volume_1d)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_1d[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_1d[0]
    vol_spike = volume_1d > vol_ma_20 * 1.5
    
    # === Align indicators to 1d timeframe (prices is already 1d) ===
    # Since prices is 1d timeframe, we can use values directly
    rsi_1d_aligned = rsi_1d
    bb_width_percentile_aligned = bb_width_percentile
    upper_bb_aligned = upper_bb
    lower_bb_aligned = lower_bb
    vol_spike_aligned = vol_spike
    atr_aligned = atr
    sma_20_aligned = sma_20
    
    # === Weekly trend filter (1w) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend
    ema_34_1w = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if i < 34:
            if i == 0:
                ema_34_1w[i] = close_1w[i]
            else:
                ema_34_1w[i] = (ema_34_1w[i-1] * i + close_1w[i]) / (i + 1)
        else:
            ema_34_1w[i] = (ema_34_1w[i-1] * 33 + close_1w[i]) / 34
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Market regime: low volatility squeeze (BB Width < 20th percentile)
        is_squeeze = bb_width_percentile_aligned[i] < 20
        
        # Entry logic: only enter when flat AND volume spike
        if position == 0:
            # Long: RSI < 30 (oversold) + BB squeeze + price near lower BB + volume spike + price above weekly EMA34
            if (rsi_1d_aligned[i] < 30 and 
                is_squeeze and 
                close[i] <= lower_bb_aligned[i] * 1.02 and  # within 2% of lower BB
                vol_spike_aligned[i] and
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI > 70 (overbought) + BB squeeze + price near upper BB + volume spike + price below weekly EMA34
            elif (rsi_1d_aligned[i] > 70 and 
                  is_squeeze and 
                  close[i] >= upper_bb_aligned[i] * 0.98 and  # within 2% of upper BB
                  vol_spike_aligned[i] and
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI crosses above 50 OR ATR-based stop (2*ATR from entry) OR price reaches upper BB
            # Simplified: exit when RSI > 50 or price > upper BB
            if (rsi_1d_aligned[i] > 50 or 
                close[i] >= upper_bb_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 50 OR ATR-based stop OR price reaches lower BB
            # Simplified: exit when RSI < 50 or price < lower BB
            if (rsi_1d_aligned[i] < 50 or 
                close[i] <= lower_bb_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_BB_Squeeze_RSI_VolumeSpike_WeeklyTrend"
timeframe = "1d"
leverage = 1.0