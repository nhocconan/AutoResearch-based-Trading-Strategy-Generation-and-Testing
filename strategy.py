#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Williams %R on 1-day timeframe with volume confirmation and ATR-based stop.
- Williams %R measures momentum overbought/oversold levels on daily timeframe
- Long when Williams %R crosses above -80 (oversold) with volume > 1.5x 20-period volume MA
- Short when Williams %R crosses below -20 (overbought) with volume > 1.5x 20-period volume MA
- Exit when Williams %R crosses back through -50 (momentum midpoint)
- Uses ATR(14) for dynamic stoploss: exit when price moves against position by 2x ATR
- Position size 0.25 to manage drawdown in volatile markets
- Designed for 4h timeframe with daily momentum filter to reduce whipsaw
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on 1-day data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_1d['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(50).values  # neutral when undefined
    
    # Align Williams %R to 4h timeframe (with 1-bar delay for completed daily bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: 20-period volume MA on 4h data
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # ATR for dynamic stoploss
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = tr1.iloc[0]  # first period
    tr3.iloc[0] = tr1.iloc[0]
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = max(20, 14)  # warmup for volume MA and Williams %R
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(williams_r_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        wr = williams_r_aligned[i]
        atr_val = atr[i]
        
        # Dynamic stoploss based on ATR
        if position == 1 and price < entry_price - 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Look for Williams %R signals with volume confirmation
            # Long: Williams %R crosses above -80 from below (bullish momentum)
            if i > start_idx and williams_r_aligned[i-1] <= -80 and wr > -80 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Williams %R crosses below -20 from above (bearish momentum)
            elif i > start_idx and williams_r_aligned[i-1] >= -20 and wr < -20 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit long when Williams %R crosses below -50 (momentum fading)
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Williams %R crosses above -50 (momentum fading)
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0