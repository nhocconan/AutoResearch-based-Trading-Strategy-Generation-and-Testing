#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h/1d Camarilla pivot breakout with volume confirmation and session filter
# - Long when price breaks above Camarilla H3 level AND 4h HMA(21) rising AND volume > 1.3x 20-period average AND session 08-20 UTC
# - Short when price breaks below Camarilla L3 level AND 4h HMA(21) falling AND volume > 1.3x 20-period average AND session 08-20 UTC
# - Exit when price returns to Camarilla pivot point (PP) or opposite breakout occurs
# - Uses discrete position sizing 0.20 to limit fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Camarilla pivots provide precise intraday support/resistance levels
# - 4h HMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false breakouts
# - Session filter avoids low-volume off-hours noise

name = "1h_4h_1d_camarilla_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 21 or len(df_1d) < 1:
        return np.zeros(n)
    
    # Pre-compute 1h data arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Pre-compute 1h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Pre-compute 4h HMA(21) for trend filter
    close_4h = df_4h['close'].values
    
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, n):
        if len(arr) < n:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, n + 1)
        return np.convolve(arr, weights[::-1], mode='valid') / weights.sum()
    
    n_hma = 21
    half_n = n_hma // 2
    sqrt_n = int(np.sqrt(n_hma))
    
    wma_half = wma(close_4h, half_n)
    wma_full = wma(close_4h, n_hma)
    
    wma_half_padded = np.full_like(close_4h, np.nan)
    wma_full_padded = np.full_like(close_4h, np.nan)
    
    if len(wma_half) > 0:
        wma_half_padded[half_n-1:half_n-1+len(wma_half)] = wma_half
    if len(wma_full) > 0:
        wma_full_padded[n_hma-1:n_hma-1+len(wma_full)] = wma_full
    
    diff = 2 * wma_half_padded - wma_full_padded
    wma_diff = wma(diff, sqrt_n)
    wma_diff_padded = np.full_like(close_4h, np.nan)
    if len(wma_diff) > 0:
        wma_diff_padded[sqrt_n-1:sqrt_n-1+len(wma_diff)] = wma_diff
    
    hma_4h = wma_diff_padded
    
    # HMA slope (rising/falling)
    hma_slope = np.diff(hma_4h, prepend=np.nan)
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    # Align 4h HMA to 1h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_4h, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_4h, hma_falling)
    
    # Pre-compute 1d Camarilla pivots (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: PP = (H+L+C)/3, Range = H-L
    # H4 = PP + Range * 1.1/2, L4 = PP - Range * 1.1/2
    # H3 = PP + Range * 1.1/4, L3 = PP - Range * 1.1/4
    # H2 = PP + Range * 1.1/6, L2 = PP - Range * 1.1/6
    # H1 = PP + Range * 1.1/12, L1 = PP - Range * 1.1/12
    pp = (high_1d + low_1d + close_1d) / 3
    rng = high_1d - low_1d
    
    h3 = pp + rng * 1.1 / 4
    l3 = pp - rng * 1.1 / 4
    h4 = pp + rng * 1.1 / 2
    l4 = pp - rng * 1.1 / 2
    
    # Align 1d Camarilla levels to 1h timeframe (with 1-bar delay for completed day)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Session filter check
        in_session = session_filter[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND 4h HMA rising AND volume spike AND in session
            if (close[i] > h3_aligned[i] and 
                hma_rising_aligned[i] and 
                volume_spike[i] and 
                in_session):
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below L3 AND 4h HMA falling AND volume spike AND in session
            elif (close[i] < l3_aligned[i] and 
                  hma_falling_aligned[i] and 
                  volume_spike[i] and 
                  in_session):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to pivot point OR opposite breakout occurs
            exit_long = (position == 1 and 
                        (close[i] < pp_aligned[i] or close[i] < l3_aligned[i]))
            exit_short = (position == -1 and 
                         (close[i] > pp_aligned[i] or close[i] > h3_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals