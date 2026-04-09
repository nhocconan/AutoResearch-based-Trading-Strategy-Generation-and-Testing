#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h/1d regime filter
# - Uses 6h Williams %R(14) for overbought/oversold conditions
# - Uses 12h ADX(14) to filter trending vs ranging markets (ADX < 25 = range)
# - Uses 1d RSI(14) for additional overbought/oversold confirmation
# - Enters mean reversion trades when Williams %R reaches extremes in ranging markets
# - Exit when price returns to 6h EMA(21) or regime shifts to trending
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to avoid fee drag
# - Williams %R is effective in ranging markets which occur frequently in bear/range regimes like 2025+

name = "6h_12h_1d_williamsr_adx_rsi_meanrev_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 12h ADX(14) for regime filter (trending vs ranging)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DM and -DM
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth the DM values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr_12h
    minus_di = 100 * minus_dm_smooth / atr_12h
    
    # Calculate DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx_12h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h ADX to 6h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 1d RSI(14) for overbought/oversold
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 6h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 6h Williams %R(14) for mean reversion signals
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high - lowest_low) != 0,
                          ((highest_high - close_6h) / (highest_high - lowest_low)) * -100,
                          -50)  # neutral when range is zero
    
    # 6h EMA(21) for exit signal
    ema_21 = pd.Series(close_6h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or
            np.isnan(adx_12h_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(ema_21[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in ranging markets (ADX < 25)
        if adx_12h_aligned[i] >= 25:
            # In trending market, exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion or regime change
            if close_6h[i] >= ema_21[i]:  # Return to mean
                position = 0
                signals[i] = 0.0
            elif adx_12h_aligned[i] >= 25:  # Regime shifted to trending
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: mean reversion or regime change
            if close_6h[i] <= ema_21[i]:  # Return to mean
                position = 0
                signals[i] = 0.0
            elif adx_12h_aligned[i] >= 25:  # Regime shifted to trending
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries in ranging market
            if (williams_r[i] <= -80 and  # Oversold
                rsi_1d_aligned[i] < 30 and  # Additional 1d oversold confirmation
                adx_12h_aligned[i] < 25):   # Confirmed ranging market
                position = 1
                signals[i] = 0.25
            elif (williams_r[i] >= -20 and  # Overbought
                  rsi_1d_aligned[i] > 70 and  # Additional 1d overbought confirmation
                  adx_12h_aligned[i] < 25):   # Confirmed ranging market
                position = -1
                signals[i] = -0.25
    
    return signals