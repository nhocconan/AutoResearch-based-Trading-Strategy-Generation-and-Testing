#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 trades over 4 years (12-37/year).
# Works in bull markets (breakouts continue trend) and bear markets (breakouts against trend filtered out).
# Added ATR-based stoploss to control drawdown.
# Focus on BTC/ETH as primary symbols with SOL as secondary validation.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla levels (R3, S3) from previous day
    # R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low)
    # Use 1d data to calculate previous day's levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 1d OHLC
    camarilla_high = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low'])
    camarilla_low = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low'])
    
    # Align to 4h timeframe (use previous day's levels)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high.values)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low.values)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 20, 34, 14, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_camarilla_high = camarilla_high_aligned[i]
        curr_camarilla_low = camarilla_low_aligned[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Camarilla break and 1d trend filter
            if curr_volume_spike:
                # Bullish: Close breaks above Camarilla R3 + price above 1d EMA34
                if curr_close > curr_camarilla_high and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below Camarilla S3 + price below 1d EMA34
                elif curr_close < curr_camarilla_low and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR close drops below Camarilla S3 OR loses 1d trend
            if curr_low <= stop_loss or curr_close < curr_camarilla_low or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR close rises above Camarilla R3 OR loses 1d trend
            if curr_high >= stop_loss or curr_close > curr_camarilla_high or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals