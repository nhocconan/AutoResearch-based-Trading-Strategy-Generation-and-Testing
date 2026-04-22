#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1-day ADX regime filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. ADX determines trend strength.
# In trending markets (ADX > 25): fade extreme Williams %R readings (mean reversion).
# In ranging markets (ADX <= 25): trade Williams %R reversals from extreme levels.
# Volume confirmation ensures institutional participation. Designed for 6BTC/ETH performance in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for ADX and Williams %R calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement for ADX
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth TR, +DM, -DM for ADX (14-period)
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI+ and DI- for ADX
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    plus_di = np.where(tr_smooth == 0, 0, plus_di)
    minus_di = np.where(tr_smooth == 0, 0, minus_di)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align ADX and Williams %R to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 6-period RSI for entry timing on 6h data
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=6, min_periods=6).mean().values
    avg_loss = pd.Series(loss).rolling(window=6, min_periods=6).mean().values
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate average volume for volume confirmation
    volume = prices['volume'].values
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_6[i]
        adx_val = adx_aligned[i]
        williams_r_val = williams_r_aligned[i]
        rsi_val = rsi[i]
        
        # Volume filter: current volume > 1.2 * 6-period average
        vol_confirm = vol > 1.2 * vol_ma
        
        # Regime detection: ADX > 25 = trending, ADX <= 25 = ranging
        is_trending = adx_val > 25
        is_ranging = adx_val <= 25
        
        if position == 0:
            if is_trending:
                # Trending regime: fade extreme Williams %R (mean reversion)
                if williams_r_val <= -80 and rsi_val < 30 and vol_confirm:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif williams_r_val >= -20 and rsi_val > 70 and vol_confirm:  # Overbought
                    signals[i] = -0.25
                    position = -1
            else:  # ranging regime
                # Ranging regime: trade Williams %R reversals from extremes
                if williams_r_val <= -90 and rsi_val < 20 and vol_confirm:  # Deep oversold
                    signals[i] = 0.25
                    position = 1
                elif williams_r_val >= -10 and rsi_val > 80 and vol_confirm:  # Deep overbought
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on Williams %R recovery or RSI overbought
                if williams_r_val >= -50 or rsi_val > 70:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on Williams %R recovery or RSI oversold
                if williams_r_val <= -50 or rsi_val < 30:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_ADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0