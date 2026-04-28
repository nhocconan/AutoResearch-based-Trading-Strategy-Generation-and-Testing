# 12h_1d_PV_Wave_Strategy
# Hypothesis: Price-volume wave (PV Wave) on daily chart filters noise and identifies strong directional moves.
# Uses 12h for entries aligned with PV Wave direction. Works in bull/bear by following institutional flow.
# Volume-weighted price deviation signals accumulation/distribution.
# Target: 20-50 trades/year, low turnover, high win rate via volume confirmation.

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
    
    # Get daily data for PV Wave
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price and money flow
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    raw_money_flow = typical_price * volume_1d
    
    # Calculate positive and negative money flow
    delta_tp = np.diff(typical_price, prepend=typical_price[0])
    positive_flow = np.where(delta_tp > 0, raw_money_flow, 0)
    negative_flow = np.where(delta_tp < 0, raw_money_flow, 0)
    
    # Money flow ratio and index (14-period)
    pos_sum = pd.Series(positive_flow).rolling(window=14, min_periods=14).sum()
    neg_sum = pd.Series(negative_flow).rolling(window=14, min_periods=14).sum()
    mfr = pos_sum / (neg_sum + 1e-10)
    mfi = 100 - (100 / (1 + mfr))
    
    # PV Wave: deviation of price from volume-weighted average price
    vwap = (typical_price * volume_1d).cumsum() / (volume_1d.cumsum() + 1e-10)
    pv_wave = typical_price - vwap
    
    # Smooth PV Wave with 3-period SMA
    pv_wave_smooth = pd.Series(pv_wave).rolling(window=3, min_periods=3).mean().values
    
    # Align daily indicators to 12h timeframe
    pv_wave_aligned = align_htf_to_ltf(prices, df_1d, pv_wave_smooth)
    mfi_aligned = align_htf_to_ltf(prices, df_1d, mfi.values)
    
    # Calculate 12h RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(pv_wave_aligned[i]) or 
            np.isnan(mfi_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # PV Wave signals: zero-cross with MFI confirmation
        pv_bullish = pv_wave_aligned[i] > 0 and pv_wave_aligned[i-1] <= 0
        pv_bearish = pv_wave_aligned[i] < 0 and pv_wave_aligned[i-1] >= 0
        
        # MFI confirms momentum (not overbought/oversold)
        mfi_bullish = mfi_aligned[i] > 50
        mfi_bearish = mfi_aligned[i] < 50
        
        # RSI for entry timing (avoid chasing)
        rsi_not_extreme = (rsi[i] > 30) and (rsi[i] < 70)
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        long_entry = pv_bullish and mfi_bullish and rsi_not_extreme and vol_confirm
        short_entry = pv_bearish and mfi_bearish and rsi_not_extreme and vol_confirm
        
        # Exit on opposite PV Wave cross
        long_exit = pv_wave_aligned[i] < 0
        short_exit = pv_wave_aligned[i] > 0
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_PV_Wave_Strategy"
timeframe = "12h"
leverage = 1.0