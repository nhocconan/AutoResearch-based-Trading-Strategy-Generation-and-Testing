#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_CAMARILLA_R1S1_BREAKOUT_VOLUME_REGIME"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # === Get 1d data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Calculate 1d CAMARILLA LEVELS ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align levels
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1h INDICATORS ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR(14) for stop loss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume condition: current volume > 1.5x 24-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=24, min_periods=24).mean().values
    vol_condition = volume > 1.5 * vol_ma
    
    # === REGIME FILTER: Choppiness Index (1d) ===
    # High CHOP = ranging (mean revert), Low CHOP = trending (follow trend)
    # We'll use CHOP > 61.8 for ranging, CHOP < 38.2 for trending
    # Calculate using 1d data
    atr_1d_list = []
    for i in range(len(high_1d)):
        if i == 0:
            tr_1d = 0
        else:
            tr_1d = max(
                abs(high_1d[i] - low_1d[i]),
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
        atr_1d_list.append(tr_1d)
    
    atr_1d_series = pd.Series(atr_1d_list)
    atr_1d_sum = atr_1d_series.rolling(window=14, min_periods=14).sum().values
    high_low_1d = np.abs(high_1d - low_1d)
    high_low_sum = pd.Series(high_low_1d).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    chop_1d = 100 * np.log10(high_low_sum / atr_1d_sum) / np.log10(14)
    chop_1d = np.where((atr_1d_sum > 0) & (high_low_sum > 0), chop_1d, 50)  # neutral when invalid
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === SESSION FILTER: 08-20 UTC ===
    hours = prices.index.hour  # already datetime64
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # Need enough data
    
    for i in range(start_idx, n):
        # Skip outside session
        if not session_mask[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        current_atr = atr[i]
        current_close = close[i]
        current_vol_cond = vol_condition[i]
        chop = chop_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1) or np.isnan(s1) or np.isnan(current_atr) or np.isnan(chop)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # === REGIME RULES ===
        # CHOP > 61.8 = ranging -> mean revert at S1/R1
        # CHOP < 38.2 = trending -> breakout
        is_ranging = chop > 61.8
        is_trending = chop < 38.2
        
        if position == 0:
            # In ranging market: buy at S1, sell at R1 (mean reversion)
            if is_ranging and current_vol_cond:
                if current_close <= s1:  # bounce at support
                    signals[i] = 0.20
                    position = 1
                    entry_price = current_close
                elif current_close >= r1:  # rejection at resistance
                    signals[i] = -0.20
                    position = -1
                    entry_price = current_close
            
            # In trending market: breakout entries
            elif is_trending and current_vol_cond:
                if current_close > r1:  # bullish breakout
                    signals[i] = 0.20
                    position = 1
                    entry_price = current_close
                elif current_close < s1:  # bearish breakout
                    signals[i] = -0.20
                    position = -1
                    entry_price = current_close
        
        elif position == 1:
            # Long exit conditions
            exit_signal = False
            
            if is_ranging:
                # In ranging: take profit at R1 or stop loss
                if current_close >= r1:
                    exit_signal = True
                elif current_close < entry_price - 1.5 * current_atr:
                    exit_signal = True
            else:  # trending
                # In trending: trail with ATR or reverse at S1
                if current_close < s1:  # trend reversal
                    exit_signal = True
                elif current_close < entry_price - 2.0 * current_atr:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit conditions
            exit_signal = False
            
            if is_ranging:
                # In ranging: take profit at S1 or stop loss
                if current_close <= s1:
                    exit_signal = True
                elif current_close > entry_price + 1.5 * current_atr:
                    exit_signal = True
            else:  # trending
                # In trending: trail with ATR or reverse at R1
                if current_close > r1:  # trend reversal
                    exit_signal = True
                elif current_close > entry_price + 2.0 * current_atr:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals