#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h chart with 1d volatility contraction (BB width percentile) and expansion breakout
# Low volatility precedes expansion; breakouts with volume capture moves in both bull/bear markets
# Uses 1w EMA40 for trend filter to avoid counter-trend trades
# Target: 12-30 trades/year per symbol (50-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # BB(20,2)
    ma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / ma_20
    
    # BB width percentile (252 lookback ~1 year)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align BB width percentile to 12h
    bb_width_pct_aligned = align_htf_to_ltf(prices, df_1d, bb_width_pct)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h ATR(20) for volatility and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 12h volume ratio (current / 50-period average)
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_ratio = volume / np.where(vol_ma_50 == 0, 1, vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if NaN in critical values
        if (np.isnan(bb_width_pct_aligned[i]) or np.isnan(ema_40_1w_aligned[i]) or
            np.isnan(atr_20[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        bb_width_pct_val = bb_width_pct_aligned[i]
        ema_trend = ema_40_1w_aligned[i]
        atr = atr_20[i]
        vol_ratio_12h = vol_ratio[i]
        
        # Determine market regime from weekly trend
        uptrend = price > ema_trend
        downtrend = price < ema_trend
        
        # Volatility contraction filter: BB width in lowest 20% percentile
        vol_contract = bb_width_pct_val < 20.0
        
        # Breakout conditions with volume confirmation
        # Upper band breakout
        upper_break = price > upper_bb[-1] if i >= len(upper_bb) else False
        # Actually need current 12h bar's relation to 1d BB - need to align BB bands
        # Instead, use: price > previous 1d close + 2*1d ATR as breakout
        
        # Calculate 1d ATR for breakout threshold
        if i < len(df_1d) * 28:  # Rough alignment: 28 12h bars per 1d
            idx_1d = i // 28
            if idx_1d < len(df_1d) and idx_1d >= 20:
                # Use pre-aligned 1d BB for current 12h bar
                ma_20_val = ma_20[idx_1d] if idx_1d < len(ma_20) else ma_20[-1]
                std_20_val = std_20[idx_1d] if idx_1d < len(std_20) else std_20[-1]
                upper_bb_val = ma_20_val + 2 * std_20_val
                lower_bb_val = ma_20_val - 2 * std_20_val
                
                # Breakout from Bollinger Bands with volume
                vol_expansion = bb_width_pct_val > 80.0  # High volatility expansion
                
                if position == 0 and vol_contract and vol_expansion:
                    if uptrend and price > upper_bb_val and vol_ratio_12h > 1.5:
                        signals[i] = 0.30
                        position = 1
                    elif downtrend and price < lower_bb_val and vol_ratio_12h > 1.5:
                        signals[i] = -0.30
                        position = -1
        
        # Exit conditions
        if position == 1:
            # Exit on volatility expansion reversal or middle band
            if bb_width_pct_val < 50.0 or price < ma_20_val:  # Return to mean
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit on volatility expansion reversal or middle band
            if bb_width_pct_val < 50.0 or price > ma_20_val:  # Return to mean
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_1d_BB_Width_Expansion_Breakout_v1"
timeframe = "12h"
leverage = 1.0