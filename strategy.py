# 2025-06-24: 4h_SmartMoney_Concepts_MarketStructure_BOS_CHOCH
# Hypothesis: Market Structure Shifts (BOS/CHoCH) on 4h timeframe with 1d trend filter and volume confirmation
# Works in bull/bear by trading with higher timeframe trend and avoiding choppy markets via structure breaks
# Uses Smart Money Concepts: Break of Structure (BOS) and Change of Character (CHoCH) to identify institutional flow
# Target: 20-40 trades/year, low frequency to avoid fee drag, high win rate via institutional confirmation

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 - higher timeframe trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Swing High/Low detection for Market Structure
    # Look for pivot points: swing high = higher high surrounded by lower highs
    # swing low = lower low surrounded by higher lows
    window = 3  # 3-bar window for swing detection (1 bar each side + center)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    
    for i in range(window, n - window):
        # Swing high: current high is highest in window
        if high[i] == np.max(high[i-window:i+window+1]):
            swing_high[i] = True
        # Swing low: current low is lowest in window
        if low[i] == np.min(low[i-window:i+window+1]):
            swing_low[i] = True
    
    # Track market structure: Higher Highs (HH), Higher Lows (HL), Lower Highs (LH), Lower Lows (LL)
    # Initialize arrays
    structure_bull = np.zeros(n, dtype=bool)  # Uptrend structure (HH and HL)
    structure_bear = np.zeros(n, dtype=bool)  # Downtrend structure (LH and LL)
    
    # Track last swing points
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    last_swing_high_idx = np.full(n, -1, dtype=int)
    last_swing_low_idx = np.full(n, -1, dtype=int)
    
    # Find swing points and track structure
    for i in range(n):
        if i == 0:
            continue
            
        # Inherit previous values
        last_swing_high[i] = last_swing_high[i-1]
        last_swing_low[i] = last_swing_low[i-1]
        last_swing_high_idx[i] = last_swing_high_idx[i-1]
        last_swing_low_idx[i] = last_swing_low_idx[i-1]
        
        # Update if current bar is a swing point
        if swing_high[i]:
            last_swing_high[i] = high[i]
            last_swing_high_idx[i] = i
        if swing_low[i]:
            last_swing_low[i] = low[i]
            last_swing_low_idx[i] = i
        
        # Determine market structure based on last two swing points
        if last_swing_high_idx[i] >= window and last_swing_low_idx[i] >= window:
            # Get the two most recent swing highs and lows
            sh_idx1 = last_swing_high_idx[i]
            sl_idx1 = last_swing_low_idx[i]
            
            # Find previous swing points
            prev_sh_idx = -1
            prev_sl_idx = -1
            for j in range(max(0, sh_idx1 - 100), sh_idx1):
                if swing_high[j]:
                    prev_sh_idx = j
                    break
            for j in range(max(0, sl_idx1 - 100), sl_idx1):
                if swing_low[j]:
                    prev_sl_idx = j
                    break
            
            if prev_sh_idx != -1 and prev_sl_idx != -1:
                # Current swing points
                sh_curr = high[sh_idx1] if sh_idx1 < n else np.nan
                sl_curr = low[sl_idx1] if sl_idx1 < n else np.nan
                # Previous swing points
                sh_prev = high[prev_sh_idx] if prev_sh_idx < n else np.nan
                sl_prev = low[prev_sl_idx] if prev_sl_idx < n else np.nan
                
                # Bullish structure: Higher High and Higher Low
                if not np.isnan(sh_curr) and not np.isnan(sh_prev) and not np.isnan(sl_curr) and not np.isnan(sl_prev):
                    if sh_curr > sh_prev and sl_curr > sl_prev:
                        structure_bull[i] = True
                    # Bearish structure: Lower High and Lower Low
                    if sh_curr < sh_prev and sl_curr < sl_prev:
                        structure_bear[i] = True
    
    # Break of Structure (BOS): price breaks beyond the last swing point in direction of trend
    bos_long = np.zeros(n, dtype=bool)   # Break above last swing high in uptrend
    bos_short = np.zeros(n, dtype=bool)  # Break below last swing low in downtrend
    
    # Change of Character (CHoCH): price breaks against the trend, indicating potential reversal
    choch_long = np.zeros(n, dtype=bool)   # Break below last swing low in uptrend (potential longs)
    choch_short = np.zeros(n, dtype=bool)  # Break above last swing high in downtrend (potential shorts)
    
    for i in range(n):
        if last_swing_high_idx[i] >= 0 and not np.isnan(last_swing_high[i]):
            if close[i] > last_swing_high[i] and structure_bull[i]:
                bos_long[i] = True  # Bullish BOS: break above swing high in uptrend
            if close[i] < last_swing_low[i] and structure_bear[i]:
                bos_short[i] = True  # Bearish BOS: break below swing low in downtrend
        
        if last_swing_low_idx[i] >= 0 and not np.isnan(last_swing_low[i]):
            if close[i] < last_swing_low[i] and structure_bull[i]:
                choch_long[i] = True  # Bullish CHoCH: break below swing low in uptrend (long opportunity)
            if close[i] > last_swing_high[i] and structure_bear[i]:
                choch_short[i] = True  # Bearish CHoCH: break above swing high in downtrend (short opportunity)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    # Trend filter: price relative to 1d EMA50
    trend_filter_long = close > ema_50_1d_aligned
    trend_filter_short = close < ema_50_1d_aligned
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient warmup for swing detection
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data is invalid
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Entry logic: BOS with trend and volume, or CHoCH with volume (counter-trend but structured)
        long_entry = (bos_long[i] or choch_long[i]) and volume_surge[i] and trend_filter_long[i]
        short_entry = (bos_short[i] or choch_short[i]) and volume_surge[i] and trend_filter_short[i]
        
        # Exit logic: opposite BOS or loss of market structure
        long_exit = bos_short[i] or choch_short[i] or not trend_filter_long[i]
        short_exit = bos_long[i] or choch_long[i] or not trend_filter_short[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.30  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.30   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_SmartMoney_Concepts_MarketStructure_BOS_CHOCH"
timeframe = "4h"
leverage = 1.0