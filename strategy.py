#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ATR filter
# - Long when price breaks above H3 pivot level AND volume > 2.0x 20-period average AND 1d ATR(14) < 20-period median ATR
# - Short when price breaks below L3 pivot level AND volume > 2.0x 20-period average AND 1d ATR(14) < 20-period median ATR
# - Exit when price returns to H4/L4 pivot levels OR ATR regime shifts to high volatility
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla pivots identify intraday support/resistance with statistical validity
# - Volume confirmation ensures breakout conviction
# - ATR filter avoids choppy markets where false breakouts occur
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_camarilla_breakout_v26"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Pre-compute 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # ATR regime: low volatility when current ATR < median of last 20 ATR values
    atr_median_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).median().values
    low_vol_regime = atr_1d < atr_median_20
    
    # Pre-compute 1d Camarilla pivot levels
    # Camarilla: H4 = C + (H-L)*1.1/2, H3 = C + (H-L)*1.1/4, H2 = C + (H-L)*1.1/6, H1 = C + (H-L)*1.1/12
    #          L1 = C - (H-L)*1.1/12, L2 = C - (H-L)*1.1/6, L3 = C - (H-L)*1.1/4, L4 = C - (H-L)*1.1/2
    camarilla_high = np.zeros((len(df_1d), 4))  # H1, H2, H3, H4
    camarilla_low = np.zeros((len(df_1d), 4))   # L1, L2, L3, L4
    
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_high[i] = np.nan
            camarilla_low[i] = np.nan
            continue
        c = close_1d[i-1]  # Previous day close
        h = high_1d[i-1]   # Previous day high
        l = low_1d[i-1]    # Previous day low
        diff = h - l
        
        camarilla_high[i, 0] = c + diff * 1.1 / 12   # H1
        camarilla_high[i, 1] = c + diff * 1.1 / 6    # H2
        camarilla_high[i, 2] = c + diff * 1.1 / 4    # H3
        camarilla_high[i, 3] = c + diff * 1.1 / 2    # H4
        
        camarilla_low[i, 0] = c - diff * 1.1 / 12    # L1
        camarilla_low[i, 1] = c - diff * 1.1 / 6     # L2
        camarilla_low[i, 2] = c - diff * 1.1 / 4     # L3
        camarilla_low[i, 3] = c - diff * 1.1 / 2     # L4
    
    # Align HTF indicators to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high[:, 2])  # H3
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high[:, 3])  # H4
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low[:, 2])   # L3
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low[:, 3])   # L4
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(low_vol_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND low volatility regime
            if (close[i] > h3_aligned[i] and 
                volume_spike[i] and 
                low_vol_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND low volatility regime
            elif (close[i] < l3_aligned[i] and 
                  volume_spike[i] and 
                  low_vol_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to H4/L4 OR volatility regime shifts to high
            exit_long = (position == 1 and (close[i] < h4_aligned[i] or not low_vol_regime_aligned[i]))
            exit_short = (position == -1 and (close[i] > l4_aligned[i] or not low_vol_regime_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals