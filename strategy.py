#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with daily volume spike and ADX trend filter.
# Long when: Price breaks above R1 (from previous day) AND daily volume > 1.5x 20-day average AND daily ADX > 25
# Short when: Price breaks below S1 (from previous day) AND daily volume > 1.5x 20-day average AND daily ADX > 25
# Exit when: Price crosses back below/above the pivot point (PP) from previous day
# Uses daily Camarilla levels for structure, volume for conviction, ADX to avoid chop.
# Target: 20-40 trades/year per symbol.
name = "4h_Camarilla_R1S1_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # R1 = C + 1.1*(H-L)/2, S1 = C - 1.1*(H-L)/2, PP = (H+L+C)/3
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    r1_1d = c_1d + 1.1 * (h_1d - l_1d) / 2
    s1_1d = c_1d - 1.1 * (h_1d - l_1d) / 2
    pp_1d = (h_1d + l_1d + c_1d) / 3
    
    # Align daily levels to 4h timeframe (available after daily candle closes)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Calculate daily ADX (trend strength)
    # TR = max(H-L, abs(H-PC), abs(L-PC))
    pc_1d = np.concatenate([c_1d[0:1], c_1d[:-1]])  # previous close
    tr1 = h_1d - l_1d
    tr2 = np.abs(h_1d - pc_1d)
    tr3 = np.abs(l_1d - pc_1d)
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM = max(H-Hprev, 0) if H-Hprev > Lprev-L else 0
    hprev_1d = np.concatenate([h_1d[0:1], h_1d[:-1]])
    lprev_1d = np.concatenate([l_1d[0:1], l_1d[:-1]])
    upmove = h_1d - hprev_1d
    downmove = lprev_1d - l_1d
    plus_dm = np.where((upmove > downmove) & (upmove > 0), upmove, 0)
    minus_dm = np.where((downmove > upmove) & (downmove > 0), downmove, 0)
    
    # Smooth TR, +DM, -DM over 14 periods
    tr_ma = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_dm_ma = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_ma = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI+ = 100 * +DM_MA / TR_MA, DI- = 100 * -DM_MA / TR_MA
    plus_di = np.where(tr_ma > 0, 100 * plus_dm_ma / tr_ma, 0)
    minus_di = np.where(tr_ma > 0, 100 * minus_dm_ma / tr_ma, 0)
    
    # DX = 100 * |DI+ - DI-| / (DI+ + DI-)
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # ADX = smoothed DX over 14 periods
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Daily volume average for confirmation
    vol_ma_20d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure daily indicators are ready (14+14+20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        pp = pp_1d_aligned[i]
        adx = adx_1d_aligned[i]
        vol_ma = vol_ma_20d_aligned[i]
        
        # Current day's volume (from 4h data, but we need daily volume)
        # Since we're using 4h data, we approximate daily volume by summing last 6 bars (4h*6=24h)
        # But simpler: use the aligned daily volume MA which represents average daily volume
        # For volume spike, we need current day's volume vs its average
        # We'll use the 4h volume and compare to daily average scaled appropriately
        # Approximate: if 4h volume > 1.5 * (daily avg volume / 6), then it's significant
        # But to keep it simple and avoid look-ahead, we'll use a volume filter on the 4h bar itself
        # using its own 20-period average (as volume tends to cluster)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
            
        vol = volume[i]
        
        if position == 0:
            # Long entry: Price breaks above R1 + volume spike + ADX > 25 (trending)
            if price > r1 and vol > 1.5 * vol_ma_20[i] and adx > 25:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S1 + volume spike + ADX > 25 (trending)
            elif price < s1 and vol > 1.5 * vol_ma_20[i] and adx > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below pivot point
            if price < pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above pivot point
            if price > pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals