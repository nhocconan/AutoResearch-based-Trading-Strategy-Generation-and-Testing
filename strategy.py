#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ATR regime filter and volume confirmation
# In high volatility regimes (ATR > 20-period ATR EMA), we trade breakouts: long on R3 breakout, short on S3 breakout.
# In low volatility regimes (ATR <= 20-period ATR EMA), we fade extremes: short near R3, long near S3.
# Volume confirmation (>1.3x 20-period volume EMA) reduces false signals.
# Designed for 12h timeframe targeting 50-150 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "12h_Camarilla_R3S3_1dATR_Regime_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR (14-period)
    tr1 = pd.Series(df_1d['high']).sub(df_1d['low'])
    tr2 = pd.Series(df_1d['high']).sub(df_1d['close'].shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).sub(df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    atr_ema_20 = pd.Series(atr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align 1d indicators to 12h timeframe
    atr_ema_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ema_20)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(atr_ema_20_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            # Determine regime: high volatility (ATR > ATR EMA) or low volatility (ATR <= ATR EMA)
            if atr_ema_20_aligned[i] > 0 and atr.iloc[i] > atr_ema_20_aligned[i]:
                # High volatility regime: trade breakouts
                if close[i] > camarilla_r3_aligned[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < camarilla_s3_aligned[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            else:
                # Low volatility regime: fade extremes (mean reversion)
                if close[i] <= camarilla_s3_aligned[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= camarilla_r3_aligned[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price retouches midpoint between R3 and S3 OR volatility drops OR volume drops
            mid = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if (close[i] <= mid or 
                (atr_ema_20_aligned[i] > 0 and atr.iloc[i] <= atr_ema_20_aligned[i]) or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches midpoint between R3 and S3 OR volatility drops OR volume drops
            mid = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if (close[i] >= mid or 
                (atr_ema_20_aligned[i] > 0 and atr.iloc[i] <= atr_ema_20_aligned[i]) or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals