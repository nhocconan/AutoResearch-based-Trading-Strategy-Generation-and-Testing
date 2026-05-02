#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses 12h timeframe for signal generation with Camarilla pivot breakouts at R3/S3 levels
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# 1d EMA34 > 1d EMA89 filters for bullish trend, < filters for bearish trend (works in both regimes)
# Discrete position sizing (0.25) balances return and risk while minimizing fee drag
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Camarilla provides mathematical support/resistance, volume confirms breakout validity
# 1d EMA trend filter ensures trades only occur in favorable higher timeframe conditions

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Calculate Camarilla pivot levels (R3, S3) from previous day
    # Typical Price = (H + L + C) / 3
    # Range = H - L
    # R3 = Close + (Range * 1.1 / 2)
    # S3 = Close - (Range * 1.1 / 2)
    typical_price = (high + low + close) / 3
    range_hl = high - low
    
    # Shift by 1 to use previous bar's data (no look-ahead)
    camarilla_r3 = close.shift(1) + (range_hl.shift(1) * 1.1 / 2)
    camarilla_s3 = close.shift(1) - (range_hl.shift(1) * 1.1 / 2)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    # Multi-timeframe: 1d EMA34 and EMA89 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align HTF EMAs to LTF (12h) with proper delay for completed 1d bar
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema89_1d)
    
    # Trend filter: bullish when EMA34 > EMA89, bearish when EMA34 < EMA89
    ema_bullish = ema34_1d_aligned > ema89_1d_aligned
    ema_bearish = ema34_1d_aligned < ema89_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3.iloc[i]) or np.isnan(camarilla_s3.iloc[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema89_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Camarilla R3 + volume confirm + 1d EMA34 > EMA89 (bullish trend)
            if close[i] > camarilla_r3.iloc[i] and volume_confirm[i] and ema_bullish[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < Camarilla S3 + volume confirm + 1d EMA34 < EMA89 (bearish trend)
            elif close[i] < camarilla_s3.iloc[i] and volume_confirm[i] and ema_bearish[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Camarilla S3 (breakdown) or 1d EMA34 < EMA89 (trend reversal)
            if close[i] < camarilla_s3.iloc[i] or ema_bearish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close > Camarilla R3 (breakout) or 1d EMA34 > EMA89 (trend reversal)
            if close[i] > camarilla_r3.iloc[i] or ema_bullish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals