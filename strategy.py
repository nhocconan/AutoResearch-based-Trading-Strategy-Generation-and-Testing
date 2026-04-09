#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h/1d regime filter and session timing
# - Uses 4h Supertrend (ATR=10, mult=3) for trend direction
# - Uses 1d RSI(14) for overbought/oversold extremes
# - Enters mean reversion trades on 1h when price touches Bollinger Bands(20,2)
# - Only takes trades aligned with 4h trend (long in uptrend, short in downtrend)
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Target: 15-30 trades/year on 1h timeframe (60-120 total over 4 years) to avoid fee drag
# - Combines trend-following (HTF) with mean reversion (LTF) for robustness in bull/bear markets

name = "1h_4h_1d_supertrend_rsi_bb_meanrev_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h Supertrend (ATR=10, mult=3) for trend direction
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(10)
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2_4h = (high_4h + low_4h) / 2
    upper_band_4h = hl2_4h + (3.0 * atr_4h)
    lower_band_4h = hl2_4h - (3.0 * atr_4h)
    
    supertrend_4h = np.full_like(close_4h, np.nan, dtype=float)
    direction_4h = np.full_like(close_4h, np.nan, dtype=float)  # 1=uptrend, -1=downtrend
    
    for i in range(10, len(close_4h)):
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 0:
            continue
            
        if i == 10:
            supertrend_4h[i] = lower_band_4h[i]
            direction_4h[i] = 1
        else:
            if close_4h[i-1] > supertrend_4h[i-1]:
                upper_band_4h[i] = min(upper_band_4h[i], upper_band_4h[i-1])
            else:
                lower_band_4h[i] = max(lower_band_4h[i], lower_band_4h[i-1])
            
            if close_4h[i] <= upper_band_4h[i]:
                supertrend_4h[i] = upper_band_4h[i]
                direction_4h[i] = -1
            else:
                supertrend_4h[i] = lower_band_4h[i]
                direction_4h[i] = 1
    
    # Align 4h Supertrend direction to 1h
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    # 1d RSI(14) for overbought/oversold
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 1h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1h Bollinger Bands(20,2) for mean reversion entries
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2.0 * std_20)
    lower_bb = sma_20 - (2.0 * std_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or
            np.isnan(direction_4h_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            std_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion or trend change
            if close[i] >= sma_20[i]:  # Return to mean
                position = 0
                signals[i] = 0.0
            elif direction_4h_aligned[i] == -1:  # 4h trend turned down
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions: mean reversion or trend change
            if close[i] <= sma_20[i]:  # Return to mean
                position = 0
                signals[i] = 0.0
            elif direction_4h_aligned[i] == 1:  # 4h trend turned up
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for mean reversion entries aligned with 4h trend
            if (close[i] <= lower_bb[i] and 
                rsi_1d_aligned[i] < 30 and  # Oversold on 1d
                direction_4h_aligned[i] == 1):  # 4h uptrend
                position = 1
                signals[i] = 0.20
            elif (close[i] >= upper_bb[i] and 
                  rsi_1d_aligned[i] > 70 and  # Overbought on 1d
                  direction_4h_aligned[i] == -1):  # 4h downtrend
                position = -1
                signals[i] = -0.20
    
    return signals