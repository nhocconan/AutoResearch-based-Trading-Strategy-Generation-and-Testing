#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for multiple indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily RSI(14) for overbought/oversold conditions
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    delta = np.concatenate([[0], delta])  # align length
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Calculate daily ATR(14) for volatility measurement
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily Bollinger Bands (20, 2) for volatility regime
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = bb_upper - bb_lower
    
    # Calculate 50-period average BB width for regime classification
    bb_width_ma_50 = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_width_ratio = bb_width / bb_width_ma_50
    
    # Align all daily indicators to 4h timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    bb_width_ratio_aligned = align_htf_to_ltf(prices, df_1d, bb_width_ratio)
    
    # 4-hour price and volume data
    price_close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume / 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(bb_width_ratio_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = price_close[i]
        rsi = rsi_14_aligned[i]
        atr = atr_14_aligned[i]
        bb_width_ratio_val = bb_width_ratio_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Regime filter: low volatility environment (BB width contraction)
        is_low_volatility = bb_width_ratio_val < 0.8
        
        if position == 0:
            # Enter long: RSI oversold in low volatility + volume confirmation
            if (rsi < 30 and is_low_volatility and vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought in low volatility + volume confirmation
            elif (rsi > 70 and is_low_volatility and vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: RSI mean reversion or volatility expansion
            if position == 1 and (rsi > 50 or bb_width_ratio_val > 1.2):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (rsi < 50 or bb_width_ratio_val > 1.2):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_RSI_MeanReversion_LowVol_Volume"
timeframe = "4h"
leverage = 1.0