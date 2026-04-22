#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h/1d regime-adaptive strategy combining ADX trend strength + RSI momentum with volume confirmation.
# Uses daily ADX to detect trending (ADX > 25) vs ranging (ADX < 20) markets with hysteresis.
# In trending regimes: RSI pullbacks to EMA with volume confirmation.
# In ranging regimes: RSI mean reversion at Bollinger Bands with volume confirmation.
# Designed for low trade frequency (<30/year) with disciplined entries to avoid fee drag.
# Works in both bull and bear markets by adapting to regime.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for ADX and RSI (once before loop)
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
    
    # Calculate ADX components
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth TR, +DM, -DM (14-period)
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    plus_di = np.where(tr_smooth == 0, 0, plus_di)
    minus_di = np.where(tr_smooth == 0, 0, minus_di)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate RSI (14-period) on daily
    delta = pd.Series(close_1d).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands (20,2) on daily
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Calculate EMA (21) on daily
    ema_21 = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align daily indicators to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21)
    
    # Calculate 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI (14) on 1h
    delta_1h = pd.Series(close).diff().values
    gain_1h = np.where(delta_1h > 0, delta_1h, 0)
    loss_1h = np.where(delta_1h < 0, -delta_1h, 0)
    avg_gain_1h = pd.Series(gain_1h).rolling(window=14, min_periods=14).mean().values
    avg_loss_1h = pd.Series(loss_1h).rolling(window=14, min_periods=14).mean().values
    rs_1h = np.where(avg_loss_1h == 0, 0, avg_gain_1h / avg_loss_1h)
    rsi_1h = 100 - (100 / (1 + rs_1h))
    
    # Volume moving average (20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(ema_21_aligned[i]) or np.isnan(rsi_1h[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        rsi_daily = rsi_aligned[i]
        rsi_hourly = rsi_1h[i]
        bb_upper = bb_upper_aligned[i]
        bb_lower = bb_lower_aligned[i]
        ema21 = ema_21_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        # Regime detection with hysteresis
        if adx_val > 25:
            regime = 'trending'
        elif adx_val < 20:
            regime = 'ranging'
        else:
            regime = regime  # maintain previous regime
        
        if position == 0:
            if regime == 'trending':
                # Trending: RSI pullback to EMA with volume spike
                if rsi_hourly < 40 and price > ema21 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif rsi_hourly > 60 and price < ema21 and vol_spike:
                    signals[i] = -0.25
                    position = -1
            else:  # ranging
                # Ranging: RSI mean reversion at Bollinger Bands
                if rsi_hourly < 30 and price <= bb_lower and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif rsi_hourly > 70 and price >= bb_upper and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on RSI overbought or price below EMA
                if rsi_hourly > 70 or price < ema21:
                    exit_signal = True
            else:  # short position
                # Exit on RSI oversold or price above EMA
                if rsi_hourly < 30 or price > ema21:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1h_ADX_RSI_Regime_Volume"
timeframe = "1h"
leverage = 1.0