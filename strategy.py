# 1d_Adaptive_RSI_Volatility_Filter
# Hypothesis: Use RSI on daily timeframe with volatility-based thresholds to capture mean reversions in both bull and bear markets. 
# In high volatility (ATR ratio > 1.2), use wider RSI bands (20/80) for mean reversion; in low volatility (ATR ratio <= 1.2), use tighter bands (30/70).
# This adapts to market regimes, reducing false signals during strong trends and capturing reversals in ranging markets.
# Volume confirmation ensures institutional participation. Designed for 1d to keep trades low (<20/year) and avoid fee drag.

name = "1d_Adaptive_RSI_Volatility_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily ATR for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.zeros_like(close_1d)
    atr_1d[0] = np.mean(tr[:14]) if len(tr) >= 14 else np.mean(tr) if len(tr) > 0 else 0
    for i in range(1, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    # Pad to match close_1d length
    atr_1d_full = np.zeros_like(close_1d)
    atr_1d_full[1:] = atr_1d
    
    # ATR ratio: current ATR / 20-period average ATR
    atr_ma_20 = pd.Series(atr_1d_full).rolling(window=20, min_periods=20).mean().values
    atr_ratio = np.where(atr_ma_20 > 0, atr_1d_full / atr_ma_20, 1.0)
    
    # Adaptive RSI bands based on volatility regime
    # High volatility (trending): wider bands (20/80) for mean reversion on pullbacks
    # Low volatility (ranging): tighter bands (30/70) for mean reversion
    rsi_lower = np.where(atr_ratio > 1.2, 20, 30)
    rsi_upper = np.where(atr_ratio > 1.2, 80, 70)
    
    # Calculate RSI on daily close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = np.mean(gain[:14]) if len(gain) >= 14 else np.mean(gain) if len(gain) > 0 else 0
    avg_loss[0] = np.mean(loss[:14]) if len(loss) >= 14 else np.mean(loss) if len(loss) > 0 else 0
    
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI and adaptive bands to lower timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    rsi_lower_aligned = align_htf_to_ltf(prices, df_1d, rsi_lower)
    rsi_upper_aligned = align_htf_to_ltf(prices, df_1d, rsi_upper)
    
    # Volume confirmation: volume > 1.3x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(rsi_lower_aligned[i]) or 
            np.isnan(rsi_upper_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # LONG: RSI crosses above lower band with volume confirmation
            if rsi_aligned[i-1] <= rsi_lower_aligned[i-1] and rsi_aligned[i] > rsi_lower_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI crosses below upper band with volume confirmation
            elif rsi_aligned[i-1] >= rsi_upper_aligned[i-1] and rsi_aligned[i] < rsi_upper_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI reaches upper band or volatility regime shifts to high volatility
            if rsi_aligned[i] >= rsi_upper_aligned[i] or atr_ratio[i] > 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI reaches lower band or volatility regime shifts to high volatility
            if rsi_aligned[i] <= rsi_lower_aligned[i] or atr_ratio[i] > 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals