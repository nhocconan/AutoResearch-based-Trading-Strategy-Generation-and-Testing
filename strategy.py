#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data for ATR and moving averages
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR(14) from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low since no previous close
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA50 and EMA200 from daily close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align indicators to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h volume moving average (20-period)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after EMA200 warmup
        # Skip if data not ready
        if (np.isnan(atr_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(vol_ma_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        atr_val = atr_aligned[i]
        ema50_val = ema50_aligned[i]
        ema200_val = ema200_aligned[i]
        vol_ma = vol_ma_12h[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirm = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: Price > EMA50 > EMA200 (strong uptrend) + volume confirmation
            if close[i] > ema50_val and ema50_val > ema200_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price < EMA50 < EMA200 (strong downtrend) + volume confirmation
            elif close[i] < ema50_val and ema50_val < ema200_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions based on ATR-based trailing stop
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price drops below EMA50 - 1.5 * ATR
                if close[i] < ema50_val - 1.5 * atr_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price rises above EMA50 + 1.5 * ATR
                if close[i] > ema50_val + 1.5 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_EMA50_EMA200_ATR_Volume_Trend"
timeframe = "12h"
leverage = 1.0