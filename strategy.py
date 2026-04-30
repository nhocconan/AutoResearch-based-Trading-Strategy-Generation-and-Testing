#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX trend filter and volume spike confirmation
# Uses discrete sizing 0.25 to minimize fee drag. Target: 50-150 total trades over 4 years (12-37/year).
# ADX > 25 ensures we only trade in trending markets, reducing false breakouts in ranging conditions.
# Volume spike (2.0x 20-bar average) confirms institutional participation.
# Works in bull markets (breakouts with trend) and bear markets (ADX filter adapts to trend strength).
# Focus on BTC/ETH as primary symbols with proven edge from Camarilla + volume + trend confluence.

name = "12h_Camarilla_R3S3_Breakout_1dADX_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla levels from prior 12h bar
    # Camarilla: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    prior_12h_high = pd.Series(high).rolling(window=2, min_periods=2).max().shift(1).values
    prior_12h_low = pd.Series(low).rolling(window=2, min_periods=2).min().shift(1).values
    prior_12h_close = pd.Series(close).rolling(window=2, min_periods=2).last().shift(1).values
    
    camarilla_r3 = prior_12h_close + 1.1 * (prior_12h_high - prior_12h_low)
    camarilla_s3 = prior_12h_close - 1.1 * (prior_12h_high - prior_12h_low)
    
    # Calculate 1d ADX(14) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().abs() * -1
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed values
    atr_1d = tr_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di_1d = 100 * (plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    minus_di_1d = 100 * (minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    
    # ADX
    dx = (abs(plus_di_1d - minus_di_1d) / (abs(plus_di_1d + minus_di_1d))) * 100
    adx_1d = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
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
    
    start_idx = max(100, 20, 14, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_adx_1d = adx_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Camarilla break and 1d ADX > 25 (trending market)
            if curr_volume_spike and curr_adx_1d > 25:
                # Bullish: Close breaks above R3
                if curr_close > curr_r3:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below S3
                elif curr_close < curr_s3:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR close drops below S3 OR ADX falls below 20 (trend weakening)
            if curr_low <= stop_loss or curr_close < curr_s3 or curr_adx_1d < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR close rises above R3 OR ADX falls below 20 (trend weakening)
            if curr_high >= stop_loss or curr_close > curr_r3 or curr_adx_1d < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals