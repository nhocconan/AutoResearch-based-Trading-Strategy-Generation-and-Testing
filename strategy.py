#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ATR-based sizing.
    # Uses 1d Camarilla levels (H3/L3) as breakout thresholds, confirmed by 4h volume spike.
    # ATR-based position sizing scales with volatility. Discrete sizes (0.0, ±0.25) minimize fee churn.
    # Target: 75-200 trades over 4 years (19-50/year) with Sharpe > 0 on BTC/ETH/SOL.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and ATR (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (H3, L3, H4, L4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:14])  # Seed with simple average
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Camarilla levels: based on previous day's range
    camarilla_h3 = np.zeros_like(close_1d)
    camarilla_l3 = np.zeros_like(close_1d)
    camarilla_h4 = np.zeros_like(close_1d)
    camarilla_l4 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Previous day's range
        range_ = high_1d[i-1] - low_1d[i-1]
        camarilla_h3[i] = close_1d[i-1] + range_ * 1.1 / 4
        camarilla_l3[i] = close_1d[i-1] - range_ * 1.1 / 4
        camarilla_h4[i] = close_1d[i-1] + range_ * 1.1 / 2
        camarilla_l4[i] = close_1d[i-1] - range_ * 1.1 / 2
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period MA
        volume_filter = volume[i] > 1.5 * volume_ma[i]
        
        # Breakout conditions: price breaks Camarilla H3/L3 with volume
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        # Dynamic position size based on volatility (ATR/price ratio)
        atr_ratio = atr_1d_aligned[i] / close[i]
        base_size = 0.25
        vol_scaled_size = base_size * (atr_ratio / 0.01)  # Normalize to 1% ATR/price
        position_size = np.clip(vol_scaled_size, 0.0, 0.30)
        
        # Entry conditions: breakout with volume confirmation
        long_entry = long_breakout and volume_filter
        short_entry = short_breakout and volume_filter
        
        # Exit conditions: opposite breakout at H4/L4 levels (wider bands)
        long_exit = close[i] < camarilla_l4_aligned[i]  # Reverse below L4
        short_exit = close[i] > camarilla_h4_aligned[i]  # Reverse above H4
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_atr_size_v1"
timeframe = "4h"
leverage = 1.0