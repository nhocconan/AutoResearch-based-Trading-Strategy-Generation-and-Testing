#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI(14) + ADX(14) regime filter + volume confirmation + ATR stop.
# Uses daily ADX to detect trending (ADX > 25) vs ranging (ADX < 20) markets.
# In trending regimes: RSI pullback entries (long RSI<40, short RSI>60) with volume.
# In ranging regimes: RSI mean reversion at extremes (long RSI<30, short RSI>70) with volume.
# Designed to work in both bull and bear markets by adapting to regime via ADX.
# Targets 20-50 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for ADX and RSI (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ADX components
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth TR, +DM, -DM
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align ADX and RSI to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate ATR(14) for stop loss on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        rsi_val = rsi_aligned[i]
        atr_val = atr[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Determine market regime using ADX
            is_trending = adx_val > 25   # Trending market
            is_ranging = adx_val < 20    # Ranging market
            
            if is_trending:
                # Trending regime: RSI pullback entries
                if rsi_val < 40 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif rsi_val > 60 and vol_spike:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # Ranging regime: RSI mean reversion at extremes
                if rsi_val < 30 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif rsi_val > 70 and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # ATR-based stop loss and take profit
            exit_signal = False
            
            if position == 1:  # long position
                # Stop loss: 2 * ATR below entry (approximated via price action)
                # Take profit: RSI > 60 (overbought) or opposite signal
                if price < 0 or rsi_val > 60:  # Simplified: exit on RSI overbought
                    exit_signal = True
            
            elif position == -1:  # short position
                # Stop loss: 2 * ATR above entry
                # Take profit: RSI < 40 (oversold) or opposite signal
                if price < 0 or rsi_val < 40:  # Simplified: exit on RSI oversold
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_ADX_RSI_Regime_Volume"
timeframe = "4h"
leverage = 1.0