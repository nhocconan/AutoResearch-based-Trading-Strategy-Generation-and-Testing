#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Chaikin Money Flow (CMF) trend filter with 1h RSI mean reversion entries.
# Uses CMF(20) on 4h to detect institutional buying/selling pressure (bullish > 0, bearish < 0).
# In bullish 4h CMF (> 0): look for 1h RSI(14) < 30 to go long (dip buying in uptrend).
# In bearish 4h CMF (< 0): look for 1h RSI(14) > 70 to go short (sell the rally in downtrend).
# Volume confirmation: require 1h volume > 1.3x 20-period average.
# Timeframe: 1h for entries, 4h for trend filter.
# Position size: 0.20 to manage drawdown and limit trade frequency.
# Designed to work in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data for CMF trend filter ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 1h data for RSI and volume ===
    # RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume ratio on 1h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # === 4h Chaikin Money Flow (CMF) ===
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # CMF = 20-period sum of Money Flow Volume / 20-period sum of Volume
    mfm = ((close_4h - low_4h) - (high_4h - close_4h)) / (high_4h - low_4h)
    mfm = np.where((high_4h - low_4h) == 0, 0, mfm)  # avoid division by zero
    mfv = mfm * volume_4h
    cmf = pd.Series(mfv).rolling(window=20, min_periods=20).sum() / pd.Series(volume_4h).rolling(window=20, min_periods=20).sum()
    cmf = cmf.values
    
    # Align 4h CMF to 1h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_4h, cmf)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(cmf_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        cmf_val = cmf_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Dynamic stop: exit if RSI shows overextension (> 70) in uptrend
            if rsi_val > 70:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Dynamic stop: exit if RSI shows overextension (< 30) in downtrend
            if rsi_val < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            if cmf_val > 0:  # Bullish 4h trend: buy dips
                if rsi_val < 30 and vol_ratio_val > 1.3:  # Oversold with volume
                    signals[i] = 0.20
                    position = 1
                    entry_price = price
                    continue
            elif cmf_val < 0:  # Bearish 4h trend: sell rallies
                if rsi_val > 70 and vol_ratio_val > 1.3:  # Overbought with volume
                    signals[i] = -0.20
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_CMF_RSI_MeanReversion_TrendFilter_Volume_v1"
timeframe = "1h"
leverage = 1.0