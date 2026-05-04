#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# In ranging/weak trending markets (price between daily EMA34 ± 1.5*ATR): fade at Camarilla R3/S3 levels
# In strong trending markets (price outside daily EMA34 ± 1.5*ATR): breakout continuation at R4/S4
# Volume confirmation (>1.3x 20-period EMA) ensures institutional participation
# Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# BTC/ETH edge: Camarilla levels provide mathematically derived support/resistance; 
# daily EMA34+ATR regime filter avoids whipsaws; volume confirms breakout validity

name = "6h_Camarilla_R3S3_R4S4_1dEMA34_ATR_Regime_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 and ATR(14)
    close_1d = pd.Series(df_1d['close'])
    ema_34 = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # ATR calculation
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    tr1 = high_1d - low_1d
    tr2 = (high_1d - close_1d.shift(1)).abs()
    tr3 = (low_1d - close_1d.shift(1)).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla uses previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    r3 = prev_close + prev_range * 1.1 / 4
    s3 = prev_close - prev_range * 1.1 / 4
    r4 = prev_close + prev_range * 1.1 / 2
    s4 = prev_close - prev_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime detection: compare price to EMA34 ± 1.5*ATR
        upper_band = ema_34_aligned[i] + 1.5 * atr_1d_aligned[i]
        lower_band = ema_34_aligned[i] - 1.5 * atr_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            if close[i] > upper_band:
                # Strong uptrend: look for breakout at R4
                if close[i] > r4_aligned[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
            elif close[i] < lower_band:
                # Strong downtrend: look for breakdown at S4
                if close[i] < s4_aligned[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            else:
                # Ranging/weak trend: fade at R3/S3
                if close[i] < r3_aligned[i] and close[i] > s3_aligned[i]:
                    # Near R3: potential short
                    if close[i] < (r3_aligned[i] + s3_aligned[i]) / 2 and volume_confirm:
                        signals[i] = -0.25
                        position = -1
                    # Near S3: potential long
                    elif close[i] > (r3_aligned[i] + s3_aligned[i]) / 2 and volume_confirm:
                        signals[i] = 0.25
                        position = 1
        elif position == 1:
            # Exit long: price below S3 OR volume drops OR reversal signal
            if (close[i] < s3_aligned[i] or 
                volume[i] < vol_ema_20[i] or
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above R3 OR volume drops OR reversal signal
            if (close[i] > r3_aligned[i] or 
                volume[i] < vol_ema_20[i] or
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals