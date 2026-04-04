#!/usr/bin/env python3
"""
exp_6587_6h_donchian20_1w_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1d/1w pivot direction and volume confirmation.
Uses 6h primary timeframe (target: 50-150 total trades over 4 years). 1d Camarilla pivot
provides intraday structure (fade at R3/S3, breakout at R4/S4), while 1w EMA200 gives
long-term trend bias. Volume confirmation ensures breakouts have conviction.
Discrete sizing (0.25) minimizes fee churn. Includes ATR-based stoploss.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6587_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 200
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0  # Volume threshold for confirmation
SIGNAL_SIZE = 0.25      # 25% position size
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5  # Stoploss at 2.5 * ATR

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for pivots and 1w for EMA200
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA200
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=EMA_PERIOD, adjust=False).mean().values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
    #          S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+Close)/3 (typical price)
    typical_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r4 = typical_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_r3 = typical_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = typical_1d - ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s4 = typical_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align to LTF (6h) with shift(1) for completed bars only
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
            
        # Determine conditions
        # Long-term trend bias from 1w EMA200
        bullish_bias = close[i] > ema_1w_aligned[i]
        bearish_bias = close[i] < ema_1w_aligned[i]
        
        # Camarilla pivot logic:
        # Fade at R3/S3 (mean reversion)
        # Breakout continuation at R4/S4
        near_r3 = abs(close[i] - camarilla_r3_aligned[i]) < (camarilla_r4_aligned[i] - camarilla_r3_aligned[i]) * 0.1
        near_s3 = abs(close[i] - camarilla_s3_aligned[i]) < (camarilla_s3_aligned[i] - camarilla_s4_aligned[i]) * 0.1
        breakout_r4 = close[i] > camarilla_r4_aligned[i]
        breakdown_s4 = close[i] < camarilla_s4_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Donchian breakout/breakdown
        donchian_breakout = close[i] > donchian_high[i-1]
        donchian_breakdown = close[i] < donchian_low[i-1]
        
        # Enter new positions only if flat
        if position == 0:
            # Long: Donchian breakout + volume + (bullish bias OR Camarilla R4 breakout)
            if (donchian_breakout and volume_confirm and 
                (bullish_bias or breakout_r4)):
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            # Short: Donchian breakdown + volume + (bearish bias OR Camarilla S4 breakdown)
            elif (donchian_breakdown and volume_confirm and 
                  (bearish_bias or breakdown_s4)):
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals