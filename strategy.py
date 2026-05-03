#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
# Long when price breaks above Camarilla R3 in bull trend (close > 1d EMA34) with volume > 2.0x 20-period MA.
# Short when price breaks below Camarilla S3 in bear trend (close < 1d EMA34) with volume spike.
# Uses discrete position sizing (0.25) to balance return and drawdown. Camarilla levels provide
# mathematically derived support/resistance based on prior day's range, ideal for intraday reversals.
# Volume confirmation ensures institutional participation. 1d trend filter reduces whipsaw vs shorter MAs.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "4h_Camarilla_R3S3_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align EMA to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar (HLC of previous completed 1d bar)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6
    #          S4 = C - (H-L)*1.1/2, S3 = C - (H-L)*1.1/4, S2 = C - (H-L)*1.1/6
    # where C = (H+L+Close)/3 of prior day
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    
    # For each 4h bar, use the prior completed 1d bar's HLC
    for i in range(n):
        # Find index of prior completed 1d bar
        # Since df_1d is already aligned to completed bars via get_htf_data, we can use direct indexing
        # But we need to map 4h bar index to 1d bar index
        # Simpler: calculate Camarilla for each 1d bar, then align to 4h
        pass
    
    # Instead, calculate Camarilla levels for each 1d bar, then align
    if len(df_1d) >= 1:
        H_1d = df_1d['high'].values
        L_1d = df_1d['low'].values
        C_1d = df_1d['close'].values
        
        # Typical price = (H+L+C)/3
        P_1d = (H_1d + L_1d + C_1d) / 3.0
        range_1d = H_1d - L_1d
        
        # Camarilla R3 and S3
        camarilla_R3_1d = P_1d + (range_1d * 1.1 / 4.0)
        camarilla_S3_1d = P_1d - (range_1d * 1.1 / 4.0)
        
        # Align to 4h timeframe (wait for completed 1d bar)
        camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3_1d)
        camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3_1d)
    
    # Donchian channels (20-period) for exit
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        r3_level = camarilla_R3_aligned[i]
        s3_level = camarilla_S3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Camarilla breakout conditions (using current bar's levels)
        breakout_up = close_val > r3_level
        breakout_down = close_val < s3_level
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_up and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and breakout_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Camarilla S3 break OR trend reversal
            if close_val < s3_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Camarilla R3 break OR trend reversal
            if close_val > r3_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals